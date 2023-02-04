import torch
import torch.nn.functional as F

from .nerf_helpers import cumprod_exclusive
from .brdf import *

brdf_specular = specular_pipeline_render_multilight_new


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :color_channel])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values

    depth_map_dex = []

    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_radiance_field_ir(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(radiance_field[..., :color_channel])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values

    depth_map_dex = []
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_radiance_field_ir_env(
    radiance_field,
    radiance_field_env,
    radiance_field_env_jitter,
    depth_values,
    ray_directions,
    c_ray_directions,
    model_env,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3,
    idx=None,
    d_n=None,
    joint=False
):
    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = radiance_field[..., :color_channel]

    rgb_map = None
    normal_map = None
    albedo_map = None
    roughness_map = None
    normals_diff_map = None
    d_n_map = None
    albedo_smoothness_cost_map = None
    roughness_smoothness_cost_map = None
    env_rgb = torch.sigmoid(rgb)

    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    env_rgb_map = weights[..., None] * env_rgb
    env_rgb_map = env_rgb_map.sum(dim=-2)
    depth_map = weights * depth_values

    depth_map_dex = []
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    if radiance_field_env is not None:
        nlights = 1

        normal = radiance_field_env[...,:3] # bs x 64 x 3
        
        albedo = radiance_field_env[...,3][...,None]
        roughness = radiance_field_env[...,4][...,None] * 0.9 + 0.09

        albedo_jitter = radiance_field_env_jitter[...,3][...,None]
        roughness_jitter = radiance_field_env_jitter[...,4][...,None] * 0.9 + 0.09

        base_albedo = torch.maximum(albedo, albedo_jitter).clip(min=1e-6)
        difference_albedo = torch.sum(((albedo - albedo_jitter)/ base_albedo)**2, dim=-1, keepdim=True) # 1024, 128, 1

        base_roughness = torch.maximum(roughness, roughness_jitter).clip(min=1e-6)
        difference_roughness = torch.sum(((roughness - roughness_jitter)/ base_roughness)**2, dim=-1, keepdim=True)

        if joint == True:
            normal_map = torch.sum(weights[..., None] * normal, -2) # bs x 3
            albedo_map = torch.sum(weights[..., None] * albedo, -2)  # bs x 1
            roughness_map = torch.sum(weights[..., None] * roughness, -2)  # bs x 1
        else:
            normal_map = torch.sum(weights[..., None].detach() * normal, -2) # bs x 3
            albedo_map = torch.sum(weights[..., None].detach() * albedo, -2)  # bs x 1
            roughness_map = torch.sum(weights[..., None].detach() * roughness, -2)  # bs x 1

        normal_map = F.normalize(normal_map, p=2, dim=-1)

        if d_n is not None:
            d_n = d_n.reshape([*normal.shape[:2],3]).detach()

            normal_diff = torch.sum(torch.pow(normal - d_n, 2), dim=-1, keepdim=True)
            normals_diff_map = torch.sum(weights[..., None].detach() * normal_diff, -2)
            d_n_map = torch.sum(weights[..., None].detach() * d_n, -2) # bs x 3

        albedo_smoothness_cost_map = torch.sum(weights[..., None].detach() * difference_albedo, -2)  # [..., 1]
        roughness_smoothness_cost_map = torch.sum(weights[..., None].detach() * difference_roughness, -2)  # [..., 1]

        albedo_map = albedo_map.clamp(0., 1.)
        roughness_map = roughness_map.clamp(0., 1.)

        fresnel_map = torch.zeros_like(albedo_map).fill_(0.04)
        fresnel_map = fresnel_map.clamp(0, 1)

        surf2c = -ray_directions
        surf2l = -ray_directions # bs x 3
        surf2c = F.normalize(surf2c,p=2.0,dim=1)
        surf2l = F.normalize(surf2l,p=2.0,dim=1).unsqueeze(-2) # bs x 1 x 3

        cosine = torch.einsum("ijk,ik->ij",surf2l, normal_map) # (bs x 1)
        specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)
        surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular # [bs, 1, 1]
        direct_light = model_env.get_light(idx).cuda().unsqueeze(-1).unsqueeze(-1) # [bs, 1, 1]

        light_rgbs = direct_light # [bs, 1, 1]
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None]  # [bs, 1, 1]

        rgb_ir = torch.sum(light_pix_contrib, dim=1)  # [bs, 1]
        rgb_map = env_rgb_map.detach() + rgb_ir
        rgb_map = torch.clip(rgb_map,0.,1.)

    out = [rgb_map, env_rgb_map, disp_map, acc_map, weights, depth_map, sigma_a, normal_map, 
            albedo_map, roughness_map, normals_diff_map, d_n_map, 
            albedo_smoothness_cost_map, roughness_smoothness_cost_map] + depth_map_dex
    return tuple(out)

def volume_render_reflectance_field(
    reflectance_field,
    SGrender,
    sg_illumination,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3
):
    n_sample = reflectance_field.shape[1]
    n_ray = reflectance_field.shape[0]
    basecolor = reflectance_field[...,:3].reshape([-1,3])
    metallic = reflectance_field[...,3:6].reshape([-1,3])
    roughness = reflectance_field[...,6:9].reshape([-1,3])
    normal = reflectance_field[...,9:12].reshape([-1,3])
    alpha = reflectance_field[...,12:15].reshape([-1,3])
    view_dir = ray_directions[...,None,:].expand(-1,n_sample,-1).reshape([-1,3])

    output = SGrender(
                sg_illuminations=sg_illumination[None,...],
                basecolor=basecolor,
                metallic=metallic,
                roughness=roughness,
                normal=normal,
                alpha=alpha,
                view_dir=view_dir,
    )
    output = torch.mean(output, dim=-1).reshape([n_ray,n_sample,1])

    one_e_10 = torch.tensor(
        [1e10], dtype=ray_directions.dtype, device=ray_directions.device
    )
    dists = torch.cat(
        (
            depth_values[..., 1:] - depth_values[..., :-1],
            one_e_10.expand(depth_values[..., :1].shape),
        ),
        dim=-1,
    )
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = torch.sigmoid(output)
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                reflectance_field[..., 15].shape,
                dtype=reflectance_field.dtype,
                device=reflectance_field.device,
            )
            * radiance_field_noise_std
        )
    sigma_a = torch.nn.functional.relu(reflectance_field[..., 15] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    depth_map = weights * depth_values

    depth_map_dex = []
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

