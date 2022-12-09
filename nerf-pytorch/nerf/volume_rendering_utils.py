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
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
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
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
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
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
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
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

def volume_render_radiance_field_ir_env(
    radiance_field,
    radiance_field_env,
    depth_values,
    ray_directions,
    c_ray_directions,
    model_env,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3,
    idx=None
):
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
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
    env_rgb = torch.sigmoid(rgb)
    #print(combined_rgb.shape, env_rgb.shape,radiance_field.shape,radiance_field_env.shape)
    #assert 1==0
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



    env_rgb_map = weights[..., None] * env_rgb
    env_rgb_map = env_rgb_map.sum(dim=-2)
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)

    if radiance_field_env is not None:
        nlights = 1
        #rgb_ir = radiance_field_env[..., :color_channel]

        normal = radiance_field_env[...,:3] # bs x 64 x 3
        albedo = radiance_field_env[...,3][...,None]
        roughness = radiance_field_env[...,4][...,None]

        normal_map = torch.sum(weights[..., None].detach() * normal, -2) # bs x 3
        albedo_map = torch.sum(weights[..., None].detach() * albedo, -2)  # bs x 1
        roughness_map = torch.sum(weights[..., None].detach() * roughness, -2)  # bs x 1

        normal_map = F.normalize(normal_map, p=2, dim=-1)
        #print(normal_map)
        #albedo_map = albedo_map.clamp(0., 1.)
        #roughness_map = roughness_map.clamp(0., 1.)
        albedo_map = albedo_map.clamp(0., 1.)
        roughness_map = roughness_map.clamp(0., 1.)
        #print(torch.norm(normal_map,dim=-1))
        #print(torch.norm(normal_map,dim=-1).shape)


        #print(albedo_map.shape)
        #assert 1==0

        fresnel_map = torch.zeros_like(albedo_map).fill_(0.04)
        fresnel_map = fresnel_map.clamp(0, 1)
        #print(normal.shape, alpha.shape, roughness.shape)
        #assert 1==0

        #print(rgb.shape)
        #assert 1==0
        surf2c = -ray_directions
        surf2l = -ray_directions # bs x 3
        surf2c = F.normalize(surf2c,p=2.0,dim=1)
        surf2l = F.normalize(surf2l,p=2.0,dim=1).unsqueeze(-2) # bs x 1 x 3

        #print(surf2cex[1,30,:] == surf2c[1,0,:])
        #assert 1==0
        #print(torch.norm(surf2c, dim=1).shape)
        cosine = torch.einsum("ijk,ik->ij",surf2l, normal_map) # (bs x 1)
        #print(cosine.shape)
        #assert 1==0


        #print(normal_map.shape, surf2c.shape, surf2l.shape, roughness_map.shape, fresnel_map.shape)
        specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)
        surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular # [bs, 1, 1]
        #surface_brdf = surface_brdf*0. + 1.
        #print(torch.mean(albedo_map))
        #if idx == None:
        #    print(surface_brdf.shape)
        #    assert 1==0
        #print(idx.shape, idx.dtype, idx[:10,:])
        direct_light = model_env.get_light(idx).cuda().unsqueeze(-1).unsqueeze(-1) # [bs, 1, 1]
        light_rgbs = direct_light # [bs, 1, 1]
        #print(torch.max(light_rgbs))
        #print(light_rgbs.shape)
        #print(torch.min(surface_brdf), torch.max(surface_brdf))
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None]  # [bs, 1, 1]
        #print(light_pix_contrib.shape)
        rgb_ir = torch.sum(light_pix_contrib, dim=1)  # [bs, 1]
        #print(relight.shape)
        #assert 1==0

        combined_rgb = torch.sigmoid(rgb.detach())# + torch.sigmoid(rgb_ir)
        #combined_rgb = torch.clip(combined_rgb,0.,1.)

        rgb_map = (weights[..., None].detach()) * combined_rgb 
        
        #print(weights.shape)
        #assert 1==0
        #print(torch.max(albedo_map), torch.max(roughness_map))
        rgb_map = rgb_map.sum(dim=-2) + rgb_ir
        rgb_map = torch.clip(rgb_map,0.,1.)
        #print(rgb_map.shape)
        #assert 1==0

    out = [rgb_map, env_rgb_map, disp_map, acc_map, weights, depth_map, sigma_a, normal_map, 
            albedo_map, roughness_map] + depth_map_dex
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
    # TESTED
    #print(depth_values[0,:])
    #print(depth_values[..., :1].shape,depth_values.shape)
    n_sample = reflectance_field.shape[1]
    n_ray = reflectance_field.shape[0]
    basecolor = reflectance_field[...,:3].reshape([-1,3])
    metallic = reflectance_field[...,3:6].reshape([-1,3])
    roughness = reflectance_field[...,6:9].reshape([-1,3])
    normal = reflectance_field[...,9:12].reshape([-1,3])
    alpha = reflectance_field[...,12:15].reshape([-1,3])
    view_dir = ray_directions[...,None,:].expand(-1,n_sample,-1).reshape([-1,3])

    #print(view_dir.shape, ray_directions.shape)

    #print(ray_directions[1,:])
    #assert 1==0

    #print(basecolor.shape, metallic.shape, roughness.shape, normal.shape, alpha.shape, view_dir.shape, sg_illumination.shape)
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
    #print(output.shape)


    #assert 1==0

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
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(reflectance_field[..., 15] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * rgb
    rgb_map = rgb_map.sum(dim=-2)
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    depth_map_dex = []
    #print(sigma_a.shape, depth_map.shape)
    #assert 1==0
    #print(m_thres_cand)
    for m_thres in m_thres_cand:
        thres_out = (sigma_a > m_thres).type(torch.int)
        #print(torch.max(sigma_a), torch.min(sigma_a))
        depth_ind = torch.argmax(thres_out, dim=-1)
        n_ind = torch.arange(depth_ind.shape[0])
        depth_map_dex.append(depth_values[n_ind, depth_ind])
    
    depth_map = depth_map.sum(dim=-1)
    #print(depth_values.shape, sigma_a.shape, depth_ind.shape, depth_map.shape, depth_map_dex.shape)
    # depth_map = (weights * depth_values).sum(dim=-1)
    #print(weights.shape)
    acc_map = weights.sum(dim=-1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map)

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    #assert 1==0
    #print(depth_map_dex.shape)
    out = [rgb_map, disp_map, acc_map, weights, depth_map, sigma_a] + depth_map_dex
    return tuple(out)

