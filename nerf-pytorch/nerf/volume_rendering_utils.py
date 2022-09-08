import torch

from .nerf_helpers import cumprod_exclusive


def volume_render_radiance_field(
    radiance_field,
    depth_values,
    ray_directions,
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None
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

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
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
    m_thres_cand=None
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

    rgb = torch.sigmoid(radiance_field[..., :3])
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
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
    radiance_field_noise_std=0.0,
    white_background=False,
    m_thres_cand=None,
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

    rgb = radiance_field[..., :3]
    rgb_ir = radiance_field_env[..., :3]
    combined_rgb = torch.sigmoid(rgb + rgb_ir)
    #print(combined_rgb.shape, combined_rgb[0,0,:])
    #assert 1==0
    noise = 0.0
    if radiance_field_noise_std > 0.0:
        noise = (
            torch.randn(
                radiance_field[..., 3].shape,
                dtype=radiance_field.dtype,
                device=radiance_field.device,
            )
            * radiance_field_noise_std
        )
        # noise = noise.to(radiance_field)
    sigma_a = torch.nn.functional.relu(radiance_field[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

    rgb_map = weights[..., None] * combined_rgb
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
