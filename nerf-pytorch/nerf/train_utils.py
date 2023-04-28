import torch
import torch.nn.functional as F
import numpy as np
import copy

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field, \
    volume_render_radiance_field_ir, volume_render_radiance_field_ir_env,volume_render_reflectance_field


def compute_err_metric(depth_gt, depth_pred, mask):
    """
    Compute the error metrics for predicted disparity map
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param mask: Selected pixel
    :return: Error metrics
    """

    depth_abs_err = F.l1_loss(depth_pred[mask] * 1000, depth_gt[mask] * 1000, reduction='mean').item()
    depth_diff = torch.abs(depth_gt[mask] - depth_pred[mask])  # [bs, 1, H, W]
    depth_err2 = depth_diff[depth_diff > 2e-3].numel() / depth_diff.numel()
    depth_err4 = depth_diff[depth_diff > 4e-3].numel() / depth_diff.numel()
    depth_err8 = depth_diff[depth_diff > 8e-3].numel() / depth_diff.numel()

    err = {}

    err['depth_abs_err'] = depth_abs_err
    err['depth_err2'] = depth_err2
    err['depth_err4'] = depth_err4
    err['depth_err8'] = depth_err8
    return err

def gen_error_colormap_depth():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 2000./(2**10) , 49, 54, 149],
         [2000./(2**10) , 2000./(2**9) , 69, 117, 180],
         [2000./(2**9) , 2000./(2**8) , 116, 173, 209],
         [2000./(2**8), 2000./(2**7), 171, 217, 233],
         [2000./(2**7), 2000./(2**6), 224, 243, 248],
         [2000./(2**6), 2000./(2**5), 254, 224, 144],
         [2000./(2**5), 2000./(2**4), 253, 174, 97],
         [2000./(2**4), 2000./(2**3), 244, 109, 67],
         [2000./(2**3), 2000./(2**2), 215, 48, 39],
         [2000./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols

def compute_obj_err(depth_gt, depth_pred, label, mask, obj_total_num=17):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """

    obj_list = label.unique()  # TODO this will cause bug if bs > 1, currently only for testing
    #print(obj_list)
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object

    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        if obj_id >= 17:
            continue
        obj_mask = label == obj_id

        obj_depth_err = F.l1_loss(depth_gt[obj_mask * mask] * 1000, depth_pred[obj_mask * mask] * 1000,
                                  reduction='mean').item()
        obj_depth_diff = torch.abs(depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask])
        #print(obj_depth_diff.numel())
        obj_depth_err4 = obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()


        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return total_obj_depth_err, total_obj_depth_4_err, total_obj_count



def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1., dilate_radius=1):
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image[0] # [H, W, 3]

def compute_err_metric(depth_gt, depth_pred, mask):
    """
    Compute the error metrics for predicted disparity map
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param mask: Selected pixel
    :return: Error metrics
    """
    depth_abs_err = F.l1_loss(depth_pred[mask] * 1000, depth_gt[mask] * 1000, reduction='mean').item()
    depth_diff = torch.abs(depth_gt[mask] - depth_pred[mask])  # [bs, 1, H, W]
    depth_err2 = depth_diff[depth_diff > 2e-3].numel() / depth_diff.numel()
    depth_err4 = depth_diff[depth_diff > 4e-3].numel() / depth_diff.numel()
    depth_err8 = depth_diff[depth_diff > 8e-3].numel() / depth_diff.numel()
    err = {}
    err['depth_abs_err'] = depth_abs_err
    err['depth_err2'] = depth_err2
    err['depth_err4'] = depth_err4
    err['depth_err8'] = depth_err8
    return err
def gen_error_colormap_depth():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 2000./(2**10) , 49, 54, 149],
         [2000./(2**10) , 2000./(2**9) , 69, 117, 180],
         [2000./(2**9) , 2000./(2**8) , 116, 173, 209],
         [2000./(2**8), 2000./(2**7), 171, 217, 233],
         [2000./(2**7), 2000./(2**6), 224, 243, 248],
         [2000./(2**6), 2000./(2**5), 254, 224, 144],
         [2000./(2**5), 2000./(2**4), 253, 174, 97],
         [2000./(2**4), 2000./(2**3), 244, 109, 67],
         [2000./(2**3), 2000./(2**2), 215, 48, 39],
         [2000./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols
def gen_error_colormap_img():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 4./(2**10) , 49, 54, 149],
         [4./(2**10) , 4./(2**9) , 69, 117, 180],
         [4./(2**9) , 4./(2**8) , 116, 173, 209],
         [4./(2**8), 4./(2**7), 171, 217, 233],
         [4./(2**7), 4./(2**6), 224, 243, 248],
         [4./(2**6), 4./(2**5), 254, 224, 144],
         [4./(2**5), 4./(2**4), 253, 174, 97],
         [4./(2**4), 4./(2**3), 244, 109, 67],
         [4./(2**3), 4./(2**2), 215, 48, 39],
         [4./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols
def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1., dilate_radius=1):
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image[0] # [H, W, 3]

def render_error_img(R_est_tensor, R_gt_tensor, mask, abs_thres=1., dilate_radius=1):
    R_gt_np = R_gt_tensor.detach().cpu().numpy()
    R_est_np = R_est_tensor.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    H, W = R_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(R_gt_np - R_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_img()
    # create error image
    error_image = np.zeros([H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # TODO: imdilate
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image # [H, W, 3]

def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field

def run_network_ir(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    #print(pts_flat.shape, embedded.shape)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)

    preds = [network_fn(batch) for batch in batches]
    #preds = network_fn(embedded[:chunksize])
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )

    return radiance_field

#def run_network_ir_env(network_fn, pts, c_pts, ray_batch, c_ray_batch, chunksize, embed_fn, embeddirs_fn):
def run_network_ir_env(network_fn, pts, chunksize, embed_fn):
    
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    #c_pts_flat = c_pts.reshape((-1, c_pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    #print(embedded.shape)
    #assert 1==0
    #c_embedded = embed_fn(c_pts_flat)
    #embedded = c_embedded
    #embedded = torch.cat((embedded, c_embedded), dim=-1)
    #print(pts_flat.shape, embedded.shape)
    """
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

        c_viewdirs = c_ray_batch[..., None, -3:]
        c_input_dirs = c_viewdirs.expand(c_pts.shape)
        c_input_dirs_flat = c_input_dirs.reshape((-1, c_input_dirs.shape[-1]))
        c_embedded_dirs = embeddirs_fn(c_input_dirs_flat)
        embedded = torch.cat((embedded, c_embedded_dirs), dim=-1)
    """
    #print("before", pts_flat[0,:], c_pts_flat[0,:], viewdirs[0,:], c_viewdirs[0,:])
    #print(pts_flat[0,:],c_pts_flat[0,:],viewdirs[0,:], c_pts_flat.shape)
    #print(embedded.shape)
    #assert 1==0
    batches = get_minibatches(embedded, chunksize=chunksize)
    #print(batches[0].shape)
    #assert 1==0
    preds = [network_fn(batch) for batch in batches]
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(
        list(pts.shape[:-1]) + [radiance_field.shape[-1]]
    )
    return radiance_field

def run_network_ir_reflectance_env(network_fn, pts, chunksize, embed_fn, embeddirs_fn):

    pts_flat = pts.reshape((-1, pts.shape[-1]))

    embedded = embed_fn(pts_flat)
 
    batches = get_minibatches(embedded, chunksize=chunksize)
    #print(batches[0].shape)
    #assert 1==0
    preds = [network_fn(batch) for batch in batches]
    refectance_field = torch.cat(preds, dim=0)
    refectance_field = refectance_field.reshape(
        list(pts.shape[:-1]) + [refectance_field.shape[-1]]
    )
    return refectance_field


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None
):
    # TESTED
    #print(ray_batch.shape)
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        #print("enter")
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])


    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    #print(z_vals.shape)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
    )
    #print(radiance_field.shape)

    coarse_out = volume_render_radiance_field(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand
    )
    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse = coarse_out[0], coarse_out[1], coarse_out[2], coarse_out[3], coarse_out[4]
    depth_coarse_dex = list(coarse_out[6:])

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()
        #print(z_samples[0,:])

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)

        #print(z_samples[0,:])
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(
            model_fine,
            pts,
            ray_batch,
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        #rgb_fine, disp_fine, acc_fine, _, _, depth_fine_dex
        fine_out = volume_render_radiance_field(
            radiance_field,
            z_vals,
            rd,
            radiance_field_noise_std=getattr(
                options.nerf, mode
            ).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            m_thres_cand=m_thres_cand
        )
        #print(z_vals[0,:])
        rgb_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2]
        depth_fine_nerf = fine_out[4]
        alpha_fine = fine_out[5]
        #print(alpha_fine[500,:])
        depth_fine_dex = list(fine_out[6:])
        #print(depth_fine_nerf.shape, alpha_fine.shape, rgb_coarse.shape)
    #print(acc_fine.shape)
    out = [rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine_nerf, alpha_fine] + depth_fine_dex
    return tuple(out)

def predict_and_render_radiance_ir(
    ray_batch,
    model_coarse,
    model_fine,
    model_env_coarse,
    model_env_fine,
    #model_fuse,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None,
    joint=False,
    albedo_edit=None,
    roughness_edit=None,
    normal_edit=None,
    logdir=None,
    light_extrinsic=None,
    is_rgb=False,
    model_backup=None,
    gt_normal=None
):

    # TESTED
    
    #assert 1==0
    num_rays = ray_batch.shape[0]
    ro, rd, c_ro, c_rd = ray_batch[..., :3], ray_batch[..., 3:6], ray_batch[..., 6:9], ray_batch[..., 9:12]
    bounds = ray_batch[..., 12:14].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    idx = ray_batch[...,14:16].type(torch.long)
    #print(ray_batch.shape, idx.shape)
    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    #print(t_vals)
    #assert 1==0
    if not getattr(options.nerf, mode).lindisp:
        #print("enter")
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])
    

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    #print(z_vals.shape)
    #print(rd.norm(p=2,dim=-1))
    #assert 1==0
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    c_pts = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network_ir(
        model_coarse,
        pts,
        ray_batch[..., -6:-3],
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn
    )
    """
    radiance_field_env = run_network_ir_env(
        model_env_coarse,
        pts,
        c_pts,
        ray_batch[..., -6:-3],
        ray_batch[..., -3:],
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn
    )
    """
    radiance_field_env = None

    #print(radiance_field_env.shape)
    #assert 1==0

    #radiance_fuse = model_fuse(torch.cat((radiance_field[...,:1],radiance_field_env),dim=-1))
    #print(radiance_field[...,:1].shape,radiance_field_env.shape,radiance_fuse.shape)

    #assert 1==0
    '''
    coarse_out = volume_render_radiance_field_ir(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand
    )
    '''
    coarse_out = volume_render_radiance_field_ir_env(
        radiance_field,
        radiance_field_env,
        None,
        z_vals,
        ro,
        rd,
        c_rd,
        model_env_coarse,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand,
        color_channel=3 if is_rgb else 1,
        idx=idx,
        d_n=None,
        joint=joint,
        light_extrinsic=light_extrinsic
    )
    rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, weights, depth_coarse = coarse_out[0], coarse_out[1], coarse_out[2], coarse_out[3], coarse_out[4], coarse_out[5]
    #assert 1==0
    #rgb_coarse_env, disp_coarse_env, acc_coarse_env, weights_env, depth_coarse_env = \
    #    coarse_out_env[0], coarse_out_env[1], coarse_out_env[2], coarse_out_env[3], coarse_out_env[4]
    #depth_coarse_dex = list(coarse_out[7:])
    #depth_coarse_dex_env = list(coarse_out_env[6:])

    #print(torch.min(rgb_coarse), torch.max(rgb_coarse))
    #assert 1==0
    #rgb_coarse_final = torch.clip(rgb_coarse + rgb_coarse_env,0.,1.)

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:
        # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()
        #print(z_samples[0,:])

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        
        #print(z_samples[0,:])
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts_fine = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        c_pts_fine = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]
        #print(rd.shape, torch.max(rd.norm(p=2,dim=-1)))
        #assert 1==0
        radiance_field = run_network_ir(
            model_fine,
            pts_fine,
            ray_batch[..., -6:-3],
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        
        radiance_field_backup = None
        if model_backup is not None:
            radiance_field_backup = run_network_ir(
                model_backup,
                pts_fine,
                ray_batch[..., -6:-3],
                getattr(options.nerf, mode).chunksize,
                encode_position_fn,
                encode_direction_fn,
            )




        radiance_field_env = None
        radiance_field_env_jitter = None
        derived_normals = None
        if not is_rgb:
            
            if mode == 'train':
                pts_fine_grad = pts_fine.clone().detach()
                pts_fine_grad.requires_grad = True
                model_fine_temp = copy.deepcopy(model_fine)
                model_fine_temp.cuda().train()
                ray_batch_grad = ray_batch[..., -6:-3].clone().detach()
                radiance_field_grad = run_network_ir(
                    model_fine_temp,
                    pts_fine_grad,
                    ray_batch_grad,
                    getattr(options.nerf, mode).chunksize,
                    encode_position_fn,
                    encode_direction_fn,
                )
                #print(pts_fine.requires_grad)
                #assert 1==0
                sigma = torch.nn.functional.relu(radiance_field_grad[..., 1])

                d_output = torch.ones_like(sigma, requires_grad=False, device=sigma.device)
                gradients = torch.autograd.grad(
                                            outputs=sigma,
                                            inputs=pts_fine_grad,
                                            grad_outputs=d_output,
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True
                                            )[0]
                #print(gradients.shape)
                                        
                derived_normals = -F.normalize(gradients, p=2, dim=-1, eps=1e-6)
                derived_normals = derived_normals.view(-1, 3)
            
            '''
            radiance_field_env = run_network_ir_env(
                model_env_fine,
                pts_fine,
                c_pts_fine,
                ray_batch[..., -6:-3],
                ray_batch[..., -3:],
                getattr(options.nerf, mode).chunksize,
                encode_position_fn,
                encode_direction_fn,
            )
            '''
            pts_fine_jitter = pts_fine + torch.randn_like(pts_fine) * 0.01
            radiance_field_env = run_network_ir_env(
                model_env_fine,
                pts_fine,
                getattr(options.nerf, mode).chunksize,
                encode_position_fn
            )
            radiance_field_env_jitter = run_network_ir_env(
                model_env_fine,
                pts_fine_jitter,
                getattr(options.nerf, mode).chunksize,
                encode_position_fn
            )

            #radiance_fuse = model_fuse(torch.cat((radiance_field[...,:1],radiance_field_env),dim=-1))
            #rgb_fine, disp_fine, acc_fine, _, _, depth_fine_dex 
            '''
            fine_out = volume_render_radiance_field_ir(
                radiance_field,
                z_vals,
                rd,
                radiance_field_noise_std=getattr(
                    options.nerf, mode
                ).radiance_field_noise_std,
                white_background=getattr(options.nerf, mode).white_background,
                m_thres_cand=m_thres_cand
            )
            '''
        fine_out = volume_render_radiance_field_ir_env(
            radiance_field,
            radiance_field_env,
            radiance_field_env_jitter,
            z_vals,
            ro,
            rd,
            c_rd,
            model_env_fine,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            m_thres_cand=m_thres_cand,
            color_channel=3 if is_rgb else 1,
            idx=idx,
            d_n=derived_normals,
            joint=joint,
            albedo_edit=albedo_edit,
            roughness_edit=roughness_edit,
            normal_edit=normal_edit,
            mode=mode,
            logdir=logdir,
            light_extrinsic=light_extrinsic,
            radiance_backup=radiance_field_backup,
            gt_normal=gt_normal
        )
        #print(z_vals[0,:])
        rgb_fine, rgb_off_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2], fine_out[3]
        normal_fine, albedo_fine, roughness_fine = fine_out[8], fine_out[9], fine_out[10]
        depth_fine_nerf = fine_out[5]
        depth_fine_nerf_backup = fine_out[6]
        alpha_fine = fine_out[7]
        normals_diff_map, d_n_map = fine_out[11], fine_out[12]
        albedo_cost_map, roughness_cost_map, normal_cost_map = fine_out[13], fine_out[14], fine_out[15]

        #rgb_fine_env, disp_fine_env, acc_fine_env, depth_fine_env = \
        #fine_out_env[0], fine_out_env[1], fine_out_env[2], fine_out_env[4]
        #print(alpha_fine[500,:])
        depth_fine_dex = list(fine_out[16:])
        #rgb_fine_final = torch.clip(rgb_fine + rgb_fine_env,0.,1.)
        #print(depth_fine_nerf.shape, alpha_fine.shape, rgb_coarse.shape)
    #print(acc_fine.shape)
    #print(alpha_fine.shape)
    #assert 1==0
    out = [rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, \
        rgb_fine, rgb_off_fine, disp_fine, acc_fine, depth_fine_nerf, depth_fine_nerf_backup, \
        alpha_fine, normal_fine, albedo_fine, roughness_fine, normals_diff_map, 
        d_n_map, albedo_cost_map, roughness_cost_map, normal_cost_map] + depth_fine_dex
    return tuple(out)

def predict_and_render_reflectance_ir(
    ray_batch,
    sceneid,
    model_coarse,
    model_fine,
    model_env,
    SGrender,
    #model_fuse,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None
):
    # TESTED
    #print(ray_batch.shape)
    num_rays = ray_batch.shape[0]
    ro, rd, c_ro, c_rd = ray_batch[..., :3], ray_batch[..., 3:6], ray_batch[..., 6:9], ray_batch[..., 9:12]
    bounds = ray_batch[..., 12:14].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
        #print("enter")
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])
    

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    #print(z_vals.shape)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    c_pts = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]



    reflectance_field_env = run_network_ir_reflectance_env(
        model_coarse,
        pts,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn
    )

    sg_param = model_env(sceneid)

    coarse_out = volume_render_reflectance_field(
        reflectance_field_env,
        SGrender,
        sg_param,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand,
        color_channel=1
    )
    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse = coarse_out[0], coarse_out[1], coarse_out[2], coarse_out[3], coarse_out[4]
    #assert 1==0
    #rgb_coarse_env, disp_coarse_env, acc_coarse_env, weights_env, depth_coarse_env = \
    #    coarse_out_env[0], coarse_out_env[1], coarse_out_env[2], coarse_out_env[3], coarse_out_env[4]
    depth_coarse_dex = list(coarse_out[6:])
    #depth_coarse_dex_env = list(coarse_out_env[6:])

    #print(torch.min(rgb_coarse), torch.max(rgb_coarse))
    #assert 1==0
    #rgb_coarse_final = torch.clip(rgb_coarse + rgb_coarse_env,0.,1.)

    rgb_fine, disp_fine, acc_fine = None, None, None

    # rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],
        getattr(options.nerf, mode).num_fine,
        det=(getattr(options.nerf, mode).perturb == 0.0),
    )
    z_samples = z_samples.detach()
    #print(z_samples[0,:])

    z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    
    #print(z_samples[0,:])
    # pts -> (N_rays, N_samples + N_importance, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    c_pts = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]



    reflectance_field_env = run_network_ir_reflectance_env(
        model_fine,
        pts,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
    )

    #radiance_fuse = model_fuse(torch.cat((radiance_field[...,:1],radiance_field_env),dim=-1))
    #rgb_fine, disp_fine, acc_fine, _, _, depth_fine_dex 
    '''
    fine_out = volume_render_radiance_field_ir(
        radiance_field,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(
            options.nerf, mode
        ).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand
    )
    '''
    fine_out = volume_render_reflectance_field(
        reflectance_field_env,
        SGrender,
        sg_param,
        z_vals,
        rd,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand,
        color_channel=1
    )
    #print(z_vals[0,:])
    rgb_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2]
    depth_fine_nerf = fine_out[4]

    #rgb_fine_env, disp_fine_env, acc_fine_env, depth_fine_env = \
    #fine_out_env[0], fine_out_env[1], fine_out_env[2], fine_out_env[4]
    #print(alpha_fine[500,:])
    depth_fine_dex = list(fine_out[6:])
    #rgb_fine_final = torch.clip(rgb_fine + rgb_fine_env,0.,1.)
    #print(depth_fine_nerf.shape, alpha_fine.shape, rgb_coarse.shape)
    #print(acc_fine.shape)
    out = [rgb_coarse, disp_coarse, acc_coarse, \
        rgb_fine, disp_fine, acc_fine, depth_fine_nerf] + depth_fine_dex
    return tuple(out)




def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None
):
    viewdirs = None
    #print(ray_directions.shape, ray_origins.shape)

    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]]
        #print(ray_directions.shape)
        restore_shapes += [torch.Size([270,480,128])]
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    #print(len(restore_shapes), ray_directions.shape[:-1])
    if options.dataset.no_ndc is False:
        #print("no_ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand
        )
        for batch in batches
    ]
    #assert 1==0
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    #print(len(synthesized_images), len(restore_shapes))
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)

def run_one_iter_of_nerf_ir(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    model_env_coarse,
    model_env_fine,
    #model_fuse,
    ray_origins,
    ray_directions,
    cam_origins,
    cam_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None,
    idx=None,
    joint=False,
    albedo_edit=None,
    roughness_edit = None,
    normal_edit=None,
    logdir=None,
    light_extrinsic=None,
    is_rgb=False,
    model_backup=None,
    gt_normal=None
):
    viewdirs = None
    #print(ray_directions.shape, ray_origins.shape)
    
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

        cam_viewdirs = cam_directions
        cam_viewdirs = cam_viewdirs / cam_viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        cam_viewdirs = cam_viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    out_shape = ray_directions[...,0].unsqueeze(-1).shape
    if is_rgb:
        out_shape = ray_directions.shape

    restore_shapes = [
        out_shape,
        out_shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # depth_fine
        restore_shapes += [ray_directions.shape[:-1]] # depth_fine_backup
        #print(out_shape)
        #assert 1==0
        #print(ray_directions.shape)
        restore_shapes += [torch.Size([270,480,128])] # alpha_fine
        restore_shapes += [torch.Size([270,480,3])] # normal_fine
        restore_shapes += [torch.Size([270,480])] # albedo_fine
        restore_shapes += [torch.Size([270,480])] # roughness_fine
        restore_shapes += [torch.Size([270,480])] # normal_diff_map
        restore_shapes += [torch.Size([270,480,3])] # d_n_map
        restore_shapes += [torch.Size([270,480])] # albedo_cost_map
        restore_shapes += [torch.Size([270,480])] # roughness_cost_map
        restore_shapes += [torch.Size([270,480])] # normal_cost_map
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    #print(len(restore_shapes), ray_directions.shape[:-1])
    if options.dataset.no_ndc is False:
        #print("no_ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        c_ro = cam_origins.view((-1, 3))
        c_rd = cam_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, c_ro, c_rd, near, far, idx), dim=-1)
    #print(rays.shape)
    #assert 1==0
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs, cam_viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    #print(len(batches))
    #assert 1==0
    pred = [
        predict_and_render_radiance_ir(
            batch,
            model_coarse,
            model_fine,
            model_env_coarse,
            model_env_fine,
            #model_fuse,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand,
            joint=joint,
            albedo_edit=albedo_edit,
            roughness_edit=roughness_edit,
            normal_edit=normal_edit,
            logdir=logdir,
            light_extrinsic=light_extrinsic,
            is_rgb=is_rgb,
            model_backup=model_backup,
            gt_normal=gt_normal
        )
        for batch in batches
    ]
    #assert 1==0
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    #print(len(synthesized_images), len(restore_shapes))
    if mode == "validation" or mode == "test":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)

def run_one_iter_of_neRF(
    height,
    width,
    focal_length,
    sceneid,
    model_coarse,
    model_fine,
    model_env,
    SGrender,
    #model_fuse,
    ray_origins,
    ray_directions,
    cam_origins,
    cam_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None
):
    viewdirs = None
    #print(ray_directions.shape, ray_origins.shape)
    
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))

        cam_viewdirs = cam_directions
        cam_viewdirs = cam_viewdirs / cam_viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        cam_viewdirs = cam_viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    out_shape = ray_directions[...,0].unsqueeze(-1).shape
    restore_shapes = [
        out_shape,
        out_shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]]
        #print(ray_directions.shape)
        restore_shapes += [torch.Size([270,480,128])]
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    #print(len(restore_shapes), ray_directions.shape[:-1])
    if options.dataset.no_ndc is False:
        #print("no_ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        c_ro = cam_origins.view((-1, 3))
        c_rd = cam_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, c_ro, c_rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs, cam_viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_reflectance_ir(
            batch,
            sceneid,
            model_coarse,
            model_fine,
            model_env,
            SGrender,
            #model_fuse,
            options,
            mode=mode,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand
        )
        for batch in batches
    ]
    #assert 1==0
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    #print(len(synthesized_images), len(restore_shapes))
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)
