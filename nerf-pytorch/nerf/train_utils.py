import torch
import torch.nn.functional as F
import numpy as np
import copy

from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .volume_rendering_utils import volume_render_radiance_field, \
    volume_render_radiance_field_ir, volume_render_radiance_field_ir_env, volume_render_reflectance_field


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

def run_network_ir_env(network_fn, pts, chunksize, embed_fn):
    
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    batches = get_minibatches(embedded, chunksize=chunksize)
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
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )
    if not getattr(options.nerf, mode).lindisp:
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
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
    )

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

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
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
        depth_fine_nerf = fine_out[4]
        alpha_fine = fine_out[5]
        depth_fine_dex = list(fine_out[6:])

    out = [rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine_nerf, alpha_fine] + depth_fine_dex
    return tuple(out)

def predict_and_render_radiance_ir(
    ray_batch,
    model_coarse,
    model_fine,
    model_env_coarse,
    model_env_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None,
    joint=False
):

    num_rays = ray_batch.shape[0]
    ro, rd, c_ro, c_rd = ray_batch[..., :3], ray_batch[..., 3:6], ray_batch[..., 6:9], ray_batch[..., 9:12]
    bounds = ray_batch[..., 12:14].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    idx = ray_batch[...,14:16].type(torch.long)
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
    radiance_field_env = None

    coarse_out = volume_render_radiance_field_ir_env(
        radiance_field,
        radiance_field_env,
        None,
        z_vals,
        rd,
        c_rd,
        model_env_coarse,
        radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
        white_background=getattr(options.nerf, mode).white_background,
        m_thres_cand=m_thres_cand,
        color_channel=1,
        idx=idx,
        d_n=None,
        joint=joint

    )
    rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, weights, depth_coarse = coarse_out[0], coarse_out[1], coarse_out[2], coarse_out[3], coarse_out[4], coarse_out[5]

    rgb_fine, disp_fine, acc_fine = None, None, None
    if getattr(options.nerf, mode).num_fine > 0:

        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid,
            weights[..., 1:-1],
            getattr(options.nerf, mode).num_fine,
            det=(getattr(options.nerf, mode).perturb == 0.0),
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        
        # pts -> (N_rays, N_samples + N_importance, 3)
        pts_fine = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
        c_pts_fine = c_ro[..., None, :] + c_rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network_ir(
            model_fine,
            pts_fine,
            ray_batch[..., -6:-3],
            getattr(options.nerf, mode).chunksize,
            encode_position_fn,
            encode_direction_fn,
        )
        derived_normals = None
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
                                    
            derived_normals = -F.normalize(gradients, p=2, dim=-1, eps=1e-6)
            derived_normals = derived_normals.view(-1, 3)
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

        fine_out = volume_render_radiance_field_ir_env(
            radiance_field,
            radiance_field_env,
            radiance_field_env_jitter,
            z_vals,
            rd,
            c_rd,
            model_env_fine,
            radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
            white_background=getattr(options.nerf, mode).white_background,
            m_thres_cand=m_thres_cand,
            color_channel=1,
            idx=idx,
            d_n=derived_normals,
            joint=joint
        )
        rgb_fine, rgb_off_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2], fine_out[3]
        normal_fine, albedo_fine, roughness_fine = fine_out[7], fine_out[8], fine_out[9]
        depth_fine_nerf = fine_out[5]
        alpha_fine = fine_out[6]
        normals_diff_map, d_n_map = fine_out[10], fine_out[11]
        albedo_cost_map, roughness_cost_map = fine_out[12], fine_out[13]

        depth_fine_dex = list(fine_out[14:])
    out = [rgb_coarse, rgb_off_coarse, disp_coarse, acc_coarse, \
        rgb_fine, rgb_off_fine, disp_fine, acc_fine, depth_fine_nerf, \
        alpha_fine, normal_fine, albedo_fine, roughness_fine, normals_diff_map, 
        d_n_map, albedo_cost_map, roughness_cost_map] + depth_fine_dex
    return tuple(out)

def predict_and_render_reflectance_ir(
    ray_batch,
    sceneid,
    model_coarse,
    model_fine,
    model_env,
    SGrender,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    m_thres_cand=None
):
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
    
    depth_coarse_dex = list(coarse_out[6:])

    rgb_fine, disp_fine, acc_fine = None, None, None

    z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    z_samples = sample_pdf(
        z_vals_mid,
        weights[..., 1:-1],
        getattr(options.nerf, mode).num_fine,
        det=(getattr(options.nerf, mode).perturb == 0.0),
    )
    z_samples = z_samples.detach()

    z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
    
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
    rgb_fine, disp_fine, acc_fine = fine_out[0], fine_out[1], fine_out[2]
    depth_fine_nerf = fine_out[4]

    depth_fine_dex = list(fine_out[6:])
    out = [rgb_coarse, disp_coarse, acc_coarse, \
        rgb_fine, disp_fine, acc_fine, depth_fine_nerf] + depth_fine_dex
    return tuple(out)
