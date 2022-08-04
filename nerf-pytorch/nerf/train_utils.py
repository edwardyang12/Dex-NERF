import torch
import torch.nn.functional as F
import numpy as np

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field

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
    depth_coarse_dex = list(coarse_out[5:])

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
        rgb_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2]
        depth_fine_dex = list(fine_out[5:])
    #print(acc_fine.shape)
    out = [rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine] + depth_fine_dex
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
