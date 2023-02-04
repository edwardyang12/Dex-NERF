import torch
import torch.nn.functional as F
import numpy as np
import copy

from .train_utils import *
from .nerf_helpers import get_minibatches, ndc_rays

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
        restore_shapes += [torch.Size([270,480,128])]
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    if options.dataset.no_ndc is False:
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
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
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
    joint=False
):
    viewdirs = None
    
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
        restore_shapes += [torch.Size([270,480,128])]
        restore_shapes += [torch.Size([270,480,3])]
        restore_shapes += [torch.Size([270,480])]
        restore_shapes += [torch.Size([270,480])]
        restore_shapes += [torch.Size([270,480])]
        restore_shapes += [torch.Size([270,480,3])]
        restore_shapes += [torch.Size([270,480])]
        restore_shapes += [torch.Size([270,480])]
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    if options.dataset.no_ndc is False:
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
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs, cam_viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
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
            joint=joint
        )
        for batch in batches
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
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
        restore_shapes += [torch.Size([270,480,128])]
        for i in m_thres_cand:
            restore_shapes += [ray_directions.shape[:-1]]
    if options.dataset.no_ndc is False:
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
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
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