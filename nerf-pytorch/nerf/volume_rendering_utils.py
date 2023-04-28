import torch
import torch.nn.functional as F

from .nerf_helpers import cumprod_exclusive
from .brdf import *

#brdf_specular = specular_pipeline_render_multilight_new
brdf_specular = specular_pipeline_render_multilight_new
import os


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
    radiance_field_env_jitter,
    depth_values,
    ray_origins,
    ray_directions,
    c_ray_directions,
    model_env,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
    color_channel=3,
    idx=None,
    d_n=None,
    joint=False,
    albedo_edit=None,
    roughness_edit=None,
    normal_edit=None,
    mode="train",
    logdir=None,
    light_extrinsic=None,
    radiance_backup=None,
    gt_normal=None
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
    #print(depth_values[0,:], dists[0,:])
    #assert 1==0
    #print(ray_directions[..., None, :].norm(p=2, dim=-1))
    #assert 1==0
    dists = dists * ray_directions[..., None, :].norm(p=2, dim=-1)

    rgb = radiance_field[..., :color_channel]
    #print(rgb[0,0,0], rgb[0,0,1], rgb[0,0,2])
    occupancy = radiance_field[..., color_channel]
    if not torch.all(~torch.isnan(rgb)):
        print("nan rgb!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    elif not torch.all(~torch.isnan(occupancy)):
        print("nan occupancy!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    rgb_map = None
    normal_map = None
    albedo_map = None
    roughness_map = None
    normals_diff_map = None
    d_n_map = None
    albedo_smoothness_cost_map = None
    roughness_smoothness_cost_map = None
    normal_smoothness_cost_map = None
    env_rgb = torch.sigmoid(rgb)

    #print(combined_rgb.shape, env_rgb.shape,radiance_field.shape,radiance_field_env.shape)
    #assert 1==0
    noise = 0.0
  
    """
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., color_channel].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        noise = noise.to(radiance_field)
    """
    sigma_a = torch.nn.functional.relu(radiance_field[..., color_channel] + noise) 
    
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    #print(dists.shape, torch.max(ray_directions[..., None, :].norm(p=2, dim=-1)))
    #assert 1==0
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)# bs x p

    


    env_rgb_map = weights[..., None] * env_rgb
    
    env_rgb_map = env_rgb_map.sum(dim=-2)
    #print(env_rgb_map.shape)
    #assert 1==0
    
    #print(torch.sum(torch.isnan(weights)), torch.sum(torch.isnan(env_rgb)), torch.sum(torch.isnan(env_rgb_map)))
    #print(depth_values[0,:])
    depth_map = weights * depth_values

    #######################################################################
    depth_map_backup = None
    if radiance_backup is not None:
        with torch.no_grad():
            sigma_a_b = torch.nn.functional.relu(radiance_backup[..., color_channel]) 
            alpha_b = 1.0 - torch.exp(-sigma_a_b * dists)
            weights_b = alpha_b * cumprod_exclusive(1.0 - alpha_b + 1e-10)# bs x p
            depth_map_backup = weights_b * depth_values
            depth_map_backup = depth_map_backup.detach()
            depth_map_backup = depth_map_backup.sum(dim=-1)
    
        #print(depth_map_backup.shape, depth_map.shape)
        #assert 1==0
    #######################################################################

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
        #weights_env = F.softmax(weights, dim=-1) # bs x p
    
        #print(weights.shape, depth_values.shape)
        #assert 1==0
        nlights = 1
        #rgb_ir = radiance_field_env[..., :color_channel]

        
        normal = radiance_field_env[...,:3] # bs x 64 x 3
        #print(torch.sum(normal[0,0,:]*normal[0,0,:]))
        #assert 1==0
        
        
        albedo = radiance_field_env[...,3][...,None]
        roughness = radiance_field_env[...,4][...,None] * 0.9 + 0.09
        #print(torch.all(~torch.isnan(albedo)).item(),torch.all(~torch.isnan(roughness)).item())
        normal_jitter = radiance_field_env_jitter[...,:3]
        #normal_norm = F.normalize(normal, dim=-1)
        #normal_jitter_norm = F.normalize(normal_jitter, dim=-1)
        #cos_normal_jitter = torch.abs(torch.sum(normal_norm * normal_jitter_norm, dim=-1))
        #print(cos_normal_jitter.shape, torch.min(cos_normal_jitter), torch.max(cos_normal_jitter))
        #assert 1==0
        albedo_jitter = radiance_field_env_jitter[...,3][...,None]
        roughness_jitter = radiance_field_env_jitter[...,4][...,None] * 0.9 + 0.09

        base_albedo = torch.maximum(albedo, albedo_jitter).clip(min=1e-6)
        difference_albedo = torch.sum(((albedo - albedo_jitter)/ base_albedo)**2, dim=-1, keepdim=True) # 1024, 128, 1

        base_roughness = torch.maximum(roughness, roughness_jitter).clip(min=1e-6)
        difference_roughness = torch.sum(((roughness - roughness_jitter)/ base_roughness)**2, dim=-1, keepdim=True)

        #print(difference_albedo.shape, difference_roughness.shape)

        
        if joint == True:
            normal_map = torch.sum(weights[..., None] * normal, -2) # bs x 3
            normal_map_jitter = torch.sum(weights[..., None].detach() * normal_jitter, -2) # bs x 3
            albedo_map = torch.sum(weights[..., None] * albedo, -2)  # bs x 1
            roughness_map = torch.sum(weights[..., None] * roughness, -2)  # bs x 1
        else:
            normal_map = torch.sum(weights[..., None].detach() * normal, -2) # bs x 3
            normal_map_jitter = torch.sum(weights[..., None].detach() * normal_jitter, -2) # bs x 3
            albedo_map = torch.sum(weights[..., None].detach() * albedo, -2)  # bs x 1
            roughness_map = torch.sum(weights[..., None].detach() * roughness, -2)  # bs x 1

        #print(normal_map.norm(p=2,dim=-1))
        #print(albedo_map.shape)
        #assert 1==0
        #normal_map = normal_map + 1e-10
        normal_map = F.normalize(normal_map, p=2, dim=-1) # bs x 3
        #print("sigma sum:", torch.sum(torch.sum(sigma_a, dim=-1)==0))
        #print("weight sum:" , torch.sum(torch.sum(weights, dim=-1)==0))
        #print("num 0norm:",torch.sum(normal.norm(p=2, dim=-1)==0).item())
        

        normal_map_jitter = F.normalize(normal_map_jitter, p=2, dim=-1)
        #print(gt_normal.shape, normal_map.shape)
        #assert 1==0
        if gt_normal is not None:
            assert 1==0
            normal_map = gt_normal

        if mode == "test":
            torch.save(weights.cpu(), os.path.join(logdir, "weights.pt"))
            torch.save(depth_values.cpu(), os.path.join(logdir, "depth_values.pt"))
            torch.save(radiance_field[..., color_channel].cpu(), os.path.join(logdir, "occu.pt"))
            torch.save(albedo_map.cpu(), os.path.join(logdir, "albedo_map.pt"))
            torch.save(roughness_map.cpu(), os.path.join(logdir, "roughness_map.pt"))


        if d_n is not None:
            """
            if joint == True:
                d_n = d_n.reshape([*normal.shape[:2],3])
                normal_diff = torch.sum(torch.pow(normal - d_n, 2), dim=-1, keepdim=True)
                normals_diff_map = torch.sum(weights[..., None] * normal_diff, -2)
                d_n_map = torch.sum(weights[..., None] * d_n, -2) # bs x 3
                #print(d_n.shape, normal.shape, normal_map.shape, d_n_map.shape)
                #assert 1==0
            else:
            """
            d_n = d_n.reshape([*normal.shape[:2],3]).detach()
            #view_direction = ray_directions[:,None,:].expand(-1,normal.shape[1],-1)

            #normal_direction_diff = torch.sum(view_direction * normal, dim=-1, keepdim=True)
            normal_diff = torch.sum(torch.pow(normal - d_n, 2), dim=-1, keepdim=True)
            normals_diff_map = torch.sum(weights[..., None].detach() * normal_diff, -2)
            d_n_map = torch.sum(weights[..., None].detach() * d_n, -2) # bs x 3

            #normals_diff_map = torch.sum(torch.pow(normal_map - d_n_map, 2), dim=-1, keepdim=True)

            #print("num 0:", torch.sum(normal_map.norm(p=2,dim=-1)==0))
        
        #print(normals_diff_map.shape)
        #assert 1==0
        cos_normal_jitter = torch.abs(torch.sum(normal_map * normal_map_jitter, dim=-1))
        #print(cos_normal_jitter.shape, torch.min(cos_normal_jitter), torch.max(cos_normal_jitter))
        #assert 1==0


        normal_smoothness_cost_map = (1. - cos_normal_jitter)[..., None]
        
        albedo_smoothness_cost_map = torch.sum(weights[..., None].detach() * difference_albedo, -2)  # [..., 1]
        roughness_smoothness_cost_map = torch.sum(weights[..., None].detach() * difference_roughness, -2)  # [..., 1]
        #print(albedo_smoothness_cost_map.shape, normal_smoothness_cost_map.shape, torch.min(normal_smoothness_cost_map), torch.max(normal_smoothness_cost_map))
        #assert 1==0

        #albedo_smoothness_loss = torch.mean(albedo_smoothness_cost_map)
        #print(albedo_smoothness_loss)
        #assert 1==0
        #print(albedo_smoothness_cost_map.shape, normals_diff_map.shape)
        #assert 1==0

        #print(normal_map)
        #albedo_map = albedo_map.clamp(0., 1.)
        #roughness_map = roughness_map.clamp(0., 1.)
        albedo_map = albedo_map.clamp(0., 1.)
        roughness_map = roughness_map.clamp(0., 1.)

        if mode == "test":
            if albedo_edit is not None:
                albedo_map = albedo_edit
            if roughness_edit is not None:
                roughness_map = roughness_edit
            if normal_edit is not None:
                normal_map = normal_edit
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
        rays_o = ray_origins
        rays_d = F.normalize(ray_directions,p=2.0,dim=1)
        surface_z = depth_map
        surface_xyz = rays_o + (surface_z).unsqueeze(-1) * rays_d  # [bs, 3]


        surf2c = -ray_directions
        #print(light_extrinsic)
        #assert 1==0
        if joint == True:
            direct_light, surf2l = model_env.get_light(surface_xyz, light_extrinsic) # bs x 3
        else:
            direct_light, surf2l = model_env.get_light(surface_xyz.detach(), light_extrinsic) # bs x 3
        direct_light = direct_light.unsqueeze(-1).unsqueeze(-1)
        #print(direct_light.shape, surf2l.shape)
        #assert 1==0


        surf2c = F.normalize(surf2c,p=2.0,dim=1)
        surf2l = F.normalize(surf2l,p=2.0,dim=1).unsqueeze(-2) # bs x 1 x 3

        #print(surf2cex[1,30,:] == surf2c[1,0,:])
        #assert 1==0
        #print(torch.norm(surf2c, dim=1).shape)
        cosine = torch.einsum("ijk,ik->ij",surf2l, normal_map) # (bs x 1)
        #print(cosine.shape)
        #assert 1==0


        #print(surf2l.shape, surf2c.shape, normal_map.shape, albedo_map.shape, roughness_map.shape)
        #assert 1==0
        #specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)
        
        """
        surface_brdf = brdf_specular(surf2l, surf2c, normal_map, albedo_map, roughness_map)

        if not torch.all(~torch.isnan(surface_brdf)):
            print(torch.all(~torch.isnan(normal_map)).item(),\
            torch.all(~torch.isnan(albedo_map)).item(),\
            torch.all(~torch.isnan(roughness_map)).item(),\
            torch.all(~torch.isnan(surface_brdf)).item())
        #    assert 1==0
       
        if mode == "test":
            torch.save(surface_brdf.cpu(), os.path.join(logdir, "brdf.pt"))
        #print(surface_brdf.shape)
        #assert 1==0
        #surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular # [bs, 1, 1]
        #surface_brdf = surface_brdf*0. + 1.
        #print(torch.mean(albedo_map))
        #if idx == None:
        #    print(surface_brdf.shape)
        #    assert 1==0
        #print(idx.shape, idx.dtype, idx[:10,:])
        #direct_light = model_env.get_light(idx).cuda().unsqueeze(-1).unsqueeze(-1) # [bs, 1, 1]
        light_rgbs = direct_light # [bs, 1, 1]
        #print(torch.max(light_rgbs))
        #print(light_rgbs.shape)
        #print(torch.min(surface_brdf), torch.max(surface_brdf))
        
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None]  # [bs, 1, 1]
        """
        specular = brdf_specular(normal_map, surf2c, surf2l, roughness_map, fresnel_map)
        surface_brdf = albedo_map.unsqueeze(1).expand(-1, nlights, -1) / np.pi + specular
        light_rgbs = direct_light
        light_pix_contrib = surface_brdf * light_rgbs * cosine[:, :, None]

        #print(light_pix_contrib.shape)
        rgb_ir = torch.sum(light_pix_contrib, dim=1)  # [bs, 1]
        #print(relight.shape)
        #assert 1==0

        #if joint == True:
        #    combined_rgb = torch.sigmoid(rgb)# + torch.sigmoid(rgb_ir)
            #combined_rgb = torch.clip(combined_rgb,0.,1.)

        #    rgb_map = (weights[..., None]) * combined_rgb 
        #else:

        #    combined_rgb = torch.sigmoid(rgb.detach())# + torch.sigmoid(rgb_ir)
            #combined_rgb = torch.clip(combined_rgb,0.,1.)

        #    rgb_map = (weights[..., None].detach()) * combined_rgb 
        
        #print(weights.shape)
        #assert 1==0
        #print(torch.max(albedo_map), torch.max(roughness_map))
        #if joint == True:
        #    rgb_map = env_rgb_map + rgb_ir
        #else:
        if joint == True:
            rgb_map = env_rgb_map + rgb_ir
        else:
            rgb_map = env_rgb_map.detach() + rgb_ir
        rgb_map = torch.clip(rgb_map,0.,1.)
        #if not torch.all(~torch.isnan(surface_brdf)):
        #    print("nan rgb_map!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #else:
        #    print("correct rgb_map!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(torch.sum(torch.isnan(weights)), torch.sum(torch.isnan(rgb)), torch.sum(torch.isnan(env_rgb_map)), torch.sum(torch.isnan(rgb_ir)))
        #print(rgb_map.shape)
        #assert 1==0
        #print(rgb_ir[0,0].item())
    out = [rgb_map, env_rgb_map, disp_map, acc_map, weights, depth_map, depth_map_backup, sigma_a, normal_map, 
            albedo_map, roughness_map, normals_diff_map, d_n_map, 
            albedo_smoothness_cost_map, roughness_smoothness_cost_map, normal_smoothness_cost_map] + depth_map_dex
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

