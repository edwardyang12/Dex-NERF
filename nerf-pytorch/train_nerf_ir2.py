import argparse
import glob
import os
import time
import copy

import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import torchvision.utils as vutils
from PIL import Image
import torch.nn.functional as F
import open3d as o3d

from nerf import compute_err_metric, depth_error_img, compute_obj_err, render_error_img

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, load_messytable_data,run_one_iter_of_nerf_ir)

debug_output = False

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--depth_supervise", action='store_true', default=False, help="use depth to supervise."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    parser.add_argument(
        "--sceneid",
        type=int,
        default=0,
        help="The scene id that need to train",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.

    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None

    # Load dataset
    images, poses, render_poses, hwf = None, None, None, None

    images, poses, ir_poses, render_poses, hwf, i_split, intrinsics, depths, labels, imgs_off, normals, roughness, albedo = load_messytable_data(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        debug = False,
        testskip=cfg.dataset.testskip,
        cfg=cfg,
        is_rgb=cfg.dataset.is_rgb,
        sceneid = configargs.sceneid,
        gt_brdf = False
    )

    color_ch = 3 if cfg.dataset.is_rgb else 1
    #print(imgs_off.shape)
    #assert 1==0
    #print(images.shape, i_split)
    #assert 1==0
    i_train, i_val, i_test = i_split
    H, W, _ = hwf
    H, W = int(H), int(W)
    if cfg.nerf.train.white_background:
        images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
    

    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    # Initialize a coarse-resolution model.
    model_coarse = getattr(models, cfg.models.coarse.type)(
        num_layers=cfg.models.coarse.num_layers,
        hidden_size=cfg.models.coarse.hidden_size,
        skip_connect_every=cfg.models.coarse.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.coarse.num_encoding_fn_xyz,
        num_encoding_fn_dir=cfg.models.coarse.num_encoding_fn_dir,
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        include_input_dir=cfg.models.coarse.include_input_dir,
        use_viewdirs=cfg.models.coarse.use_viewdirs,
        color_channel=color_ch
    )
    model_coarse.to(device)

    model_env_coarse = None
    model_env_fine = None
    if not cfg.dataset.is_rgb:
        ir_intrinsic = intrinsics[0,:,:].to(device)
        ir_intrinsic[:2,2] = ir_intrinsic[:2,2] * 2.
        ir_extrinsic = ir_poses[0,:,:].to(device)
        model_env_coarse = getattr(models, cfg.models.env.type)(
            num_layers=cfg.models.env.num_layers,
            hidden_size=cfg.models.env.hidden_size,
            skip_connect_every=cfg.models.env.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.env.num_encoding_fn_xyz,
            #num_encoding_fn_dir=cfg.models.env.num_encoding_fn_dir,
            include_input_xyz=cfg.models.env.include_input_xyz,
            #include_input_dir=cfg.models.env.include_input_dir,
            #use_viewdirs=cfg.models.env.use_viewdirs,
            color_channel=1,
            H = cfg.dataset.H,
            W = cfg.dataset.W,
            ir_intrinsic=ir_intrinsic,
            ir_extrinsic=ir_extrinsic,
            #ir_gt="/code/nerf-git/logs_sim_brdf_near_check/0/ir_pat.pt"
        )
        model_env_coarse.to(device)
        model_env_fine = getattr(models, cfg.models.env.type)(
            num_layers=cfg.models.env.num_layers,
            hidden_size=cfg.models.env.hidden_size,
            skip_connect_every=cfg.models.env.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.env.num_encoding_fn_xyz,
            #num_encoding_fn_dir=cfg.models.env.num_encoding_fn_dir,
            include_input_xyz=cfg.models.env.include_input_xyz,
            #include_input_dir=cfg.models.env.include_input_dir,
            #use_viewdirs=cfg.models.env.use_viewdirs,
            color_channel=1,
            H = cfg.dataset.H,
            W = cfg.dataset.W,
            ir_intrinsic=ir_intrinsic,
            ir_extrinsic=ir_extrinsic,
            #ir_gt="/code/nerf-git/logs_sim_brdf_near_check/0/ir_pat.pt"
        )
        model_env_fine.to(device)




    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    if hasattr(cfg.models, "fine"):
        model_fine = getattr(models, cfg.models.fine.type)(
            num_layers=cfg.models.fine.num_layers,
            hidden_size=cfg.models.fine.hidden_size,
            skip_connect_every=cfg.models.fine.skip_connect_every,
            num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
            num_encoding_fn_dir=cfg.models.fine.num_encoding_fn_dir,
            include_input_xyz=cfg.models.fine.include_input_xyz,
            include_input_dir=cfg.models.fine.include_input_dir,
            use_viewdirs=cfg.models.fine.use_viewdirs,
            color_channel=color_ch
        )
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    #trainable_parameters_env = list(model_env_coarse.parameters())
    

    trainable_parameters += list(model_fine.parameters())
    #trainable_parameters_env += list(model_env_fine.parameters())
    if not cfg.dataset.is_rgb:
        trainable_parameters += list(model_env_coarse.parameters())
        trainable_parameters += list(model_env_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    #optimizer_env = getattr(torch.optim, cfg.optimizer.type)(
    #    trainable_parameters_env, lr=cfg.optimizer.lr
    #)

    
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, str(configargs.sceneid))
    os.makedirs(logdir, exist_ok=True)
    
    m_thres_max = cfg.nerf.validation.m_thres
    m_thres_cand = np.arange(5,m_thres_max+5,1)
    os.makedirs(os.path.join(logdir,"pred_depth_dex"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_err_dex"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_err_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_nerf_gt"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_pcd_nerf"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_pcd_nerf_gt"), exist_ok=True)
    

    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if not cfg.dataset.is_rgb:
            model_env_coarse.load_state_dict(checkpoint["model_env_coarse_state_dict"])
            model_env_fine.load_state_dict(checkpoint["model_env_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays
    #no_ir_train = True
    #jointtrain = False
    is_joint = False
    #model_env_fine.light_attenuation_coeff.requires_grad = False

    #prev_params = list(model_env_fine.parameters())
    for param in model_coarse.parameters():
        param.requires_grad = True
    for param in model_fine.parameters():
        param.requires_grad = True
    for param in model_env_coarse.parameters():
        param.requires_grad = True
    for param in model_env_fine.parameters():
        param.requires_grad = True

    train_depth = True
    
    model_backup = None

    for i in trange(start_iter, cfg.experiment.train_iters):
        if i == cfg.experiment.jointtrain_start:
            is_joint = True
            model_backup = copy.deepcopy(model_fine)
            for param in model_backup.parameters():
                param.requires_grad = False

        if is_joint == True and i < cfg.experiment.joint_start:
            if i % cfg.experiment.swap_every == 0:
                if train_depth:
                    #print("train_depth_swap true")
                    for param in model_coarse.parameters():
                        param.requires_grad = False
                    for param in model_fine.parameters():
                        param.requires_grad = False
                    for param in model_env_coarse.parameters():
                        param.requires_grad = True
                    for param in model_env_fine.parameters():
                        param.requires_grad = True

                    model_coarse.eval()
                    model_fine.eval()
                    if not cfg.dataset.is_rgb:
                        model_env_coarse.train()
                        model_env_fine.train()

                    train_depth = False
                else:
                    #print("train_depth_swap false")
                    for param in model_coarse.parameters():
                        param.requires_grad = True
                    for param in model_fine.parameters():
                        param.requires_grad = True
                    for param in model_env_coarse.parameters():
                        param.requires_grad = False
                    for param in model_env_fine.parameters():
                        param.requires_grad = False

                    model_coarse.train()
                    model_fine.train()
                    if not cfg.dataset.is_rgb:
                        model_env_coarse.eval()
                        model_env_fine.eval()
                    
                    train_depth = True
        elif i == cfg.experiment.joint_start:
            #print("train_joint_start")
            for param in model_coarse.parameters():
                param.requires_grad = True
            for param in model_fine.parameters():
                param.requires_grad = True
            for param in model_env_coarse.parameters():
                param.requires_grad = True
            for param in model_env_fine.parameters():
                param.requires_grad = True
            
            model_env_fine.ir_pattern.requires_grad = False
            model_env_fine.static_ir_pat = True
                
            model_coarse.train()
            model_fine.train()
            if not cfg.dataset.is_rgb:
                model_env_coarse.train()
                model_env_fine.train()



        rgb_coarse, rgb_fine = None, None
        target_ray_values = None

        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        
        pose_target = poses[img_idx, :, :].to(device)
        depth_target = depths[img_idx].to(device)
        normal_target = normals[img_idx].to(device)
        if roughness is not None:
            roughness_target = roughness[img_idx].to(device)
            albedo_target = albedo[img_idx].to(device)
        #print(normal_target.shape)
        #assert 1==0
        #print(img_target.shape, depth_target.shape)
        #assert 1==0
        #print("===========================================")
        #print(pose_target)

        #print(pose_target.shape)
        intrinsic_target = intrinsics[img_idx,:,:].to(device)
        ir_extrinsic_target = ir_poses[img_idx,:,:].to(device)

        img_off_target = imgs_off[img_idx].to(device)
        #print(intrinsic_target)
        #print(img_idx)
        #print("===========================================")
        ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(H, W, focal, pose_target, intrinsic_target)
        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        coords = coords.reshape((-1, 2))
        select_inds = np.random.choice(
            coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
        )
        select_inds = coords[select_inds]
        #print(ray_origins.shape)
        #assert 1==0
        ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
        #print(ray_directions.shape)
        ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
        #print(torch.max(ray_directions.norm(p=2,dim=-1)))
        #assert 1==0
        #print(ray_directions.shape)
        # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
        #print(img_target.shape)
        cam_origins = cam_origins[select_inds[:, 0], select_inds[:, 1], :]
        #print(ray_directions.shape)
        cam_directions = cam_directions[select_inds[:, 0], select_inds[:, 1], :]
        
        target_s = img_target[select_inds[:, 0], select_inds[:, 1]] # [1080]
        #print(select_inds)
        #assert 1==0 
        #print(target_s.shape)
        #assert 1==0
        #target_env = pattern_target[select_inds[:, 0], select_inds[:, 1]]
        target_d = depth_target[select_inds[:, 0], select_inds[:, 1]]
        target_s_off = img_off_target[select_inds[:, 0], select_inds[:, 1]]
        target_n = normal_target[select_inds[:, 0], select_inds[:, 1], :]
        #znorm = target_n.norm(p=2, dim=-1) == 0
        #print((znorm == True).nonzero(as_tuple=True)[0])
        if roughness is not None:
            target_roughness = roughness_target[select_inds[:, 0], select_inds[:, 1]]
            target_albedo = albedo_target[select_inds[:, 0], select_inds[:, 1]]
        

        #print(target_s.shape)
        #assert 1==0
        then = time.time()
        #print(ray_origins.shape, ray_directions.shape)
        #print(ray_origins[-3:,:], ray_directions[-3:,:])
        #rgb_coarse, _, _, rgb_fine, _, _, _
        #print(ir_extrinsic_target)
        #assert 1==0
        #print("before, " , torch.max(ray_directions.norm(p=2, dim=-1)))

        nerf_out = run_one_iter_of_nerf_ir(
            H,
            W,
            intrinsic_target[0,0],
            model_coarse,
            model_fine,
            model_env_coarse,
            model_env_fine,
            #model_fuse,
            ray_origins,
            ray_directions,
            cam_origins.cuda(),
            cam_directions.cuda(),
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand,
            idx=select_inds,
            joint=is_joint,
            light_extrinsic=ir_extrinsic_target,
            is_rgb=cfg.dataset.is_rgb,
            model_backup=model_backup,
            #gt_normal=target_n
        )
        rgb_coarse, rgb_off_coarse, rgb_fine, rgb_off_fine = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
        depth_fine_nerf = nerf_out[8]
        depth_fine_nerf_backup = nerf_out[9]
        alpha_fine = nerf_out[10]
        normal_fine = nerf_out[11]
        #print(rgb_fine)
        albedo_fine = nerf_out[12]
        roughness_fine = nerf_out[13]
        #print(roughness_fine.shape, target_roughness.shape)
        #assert 1==0

        normals_diff_map = nerf_out[14]
        d_n_map = nerf_out[15]
        albedo_cost_map = nerf_out[16]
        
        roughness_cost_map = nerf_out[17]
        normal_cost_map = nerf_out[18]


        #rgb_coarse = torch.mean(rgb_coarse, dim=-1)
        #rgb_fine = torch.mean(rgb_fine, dim=-1)
        #print(rgb_coarse.shape, rgb_fine.shape)
        target_ray_values = target_s.unsqueeze(-1)
        target_ray_values_off = target_s_off.unsqueeze(-1)
        #print(rgb_coarse.shape, rgb_fine.shape, target_ray_values.shape)
        #assert 1==0
        
        #if i == cfg.experiment.finetune_start:
        #    no_ir_train = False
        
        
        #print(model_env_fine.light_attenuation_coeff.item())
        coarse_loss = 0.0
        #print(model_env_fine.light_attenuation_coeff.item())

        #print(target_d.shape, torch.max(target_d))
        #assert 1==0
        #img_ground_mask = (target_d > 0) & (target_d < 1.25)

        coarse_loss_off = torch.nn.functional.mse_loss(
            torch.squeeze(rgb_off_coarse), torch.squeeze(target_ray_values_off)
        )
        #print(torch.squeeze(rgb_off_fine)[500,:], torch.squeeze(target_ray_values_off)[500,:])
        #assert 1==0
        fine_loss_off = torch.nn.functional.mse_loss(
            torch.squeeze(rgb_off_fine), torch.squeeze(target_ray_values_off)
        )

        fine_loss = 0
        d_normal_loss_gt = 0
        fine_normal_loss_gt = 0
        fine_nerf_depth_loss_gt = 0
        fine_nerf_depth_loss_backup = 0
        fine_normal_loss = 0
        albedo_smoothness_loss = 0
        roughness_smoothness_loss = 0
        normal_smoothness_loss = 0
        fine_roughness_loss_gt = 0
        fine_albedo_loss_gt = 0

        """
        if not torch.all(~torch.isnan(rgb_off_fine)):
            print("nan output rgbofffine")
            return

        if not torch.all(~torch.isnan(rgb_fine)):
            print("nan output rgbfine")
            return
        """
        
        if not cfg.dataset.is_rgb:
            fine_loss = torch.nn.functional.mse_loss(
                    rgb_fine, target_ray_values
            )
            d_normal_loss_gt = torch.nn.functional.mse_loss(
                d_n_map, target_n
            )
            fine_normal_loss_gt = torch.nn.functional.mse_loss(
                normal_fine, target_n
            )
            if depth_fine_nerf_backup is not None:
                fine_nerf_depth_loss_backup = torch.nn.functional.mse_loss(
                    depth_fine_nerf, depth_fine_nerf_backup
                )
            fine_nerf_depth_loss_gt = torch.nn.functional.mse_loss(
                depth_fine_nerf, target_d
            )
            if roughness is not None:
                fine_roughness_loss_gt = torch.nn.functional.mse_loss(
                    torch.squeeze(roughness_fine), torch.squeeze(target_roughness)
                )
            if albedo is not None:
                fine_albedo_loss_gt = torch.nn.functional.mse_loss(
                    torch.squeeze(albedo_fine), torch.squeeze(target_albedo)
                )
            #print(d_n_map.shape, (target_n*(-1.)).shape)
            #assert 1==0
            fine_normal_loss = normals_diff_map.mean()

            albedo_smoothness_loss = torch.mean(albedo_cost_map)
            roughness_smoothness_loss = torch.mean(roughness_cost_map)
            normal_smoothness_loss = torch.mean(normal_cost_map)
        #print(coarse_loss_off.item(), fine_loss.item(), fine_loss_off.item(), fine_normal_loss.item(), \
        #albedo_smoothness_loss.item(), roughness_smoothness_loss.item(), normal_smoothness_loss.item())
        #print(fine_normal_loss)
        if not torch.all(~torch.isnan(fine_loss)):
            print("nan fineloss")
            return

        #if i < 10000:
        #    fine_normal_loss = fine_normal_loss*0.
        if is_joint:
            depth_rate = cfg.experiment.depth_rate
        else:
            depth_rate = 0

        loss_off = coarse_loss_off + fine_loss_off
        
        #print(fine_loss.item(), fine_normal_loss.item())
        loss_on = fine_loss + \
            cfg.experiment.normal_gt_rate * fine_normal_loss_gt + \
            cfg.experiment.normal_derived_rate * fine_normal_loss + \
            cfg.experiment.albedo_rate * albedo_smoothness_loss + \
            cfg.experiment.roughness_rate * roughness_smoothness_loss + \
            cfg.experiment.normal_rate * normal_smoothness_loss + \
            cfg.experiment.rougness_gt_rate * fine_roughness_loss_gt + \
            cfg.experiment.albedo_gt_rate * fine_albedo_loss_gt

        loss = cfg.experiment.ir_on_rate * loss_on + \
            cfg.experiment.ir_off_rate * loss_off + \
            cfg.experiment.depth_rate_backup * fine_nerf_depth_loss_backup + \
            depth_rate * fine_nerf_depth_loss_gt




        if is_joint and cfg.experiment.grad_norm_rate > 0:
            grad_params_coarse = torch.autograd.grad(loss, model_coarse.parameters(), create_graph=True)
            grad_params_fine = torch.autograd.grad(loss, model_fine.parameters(), create_graph=True)

            grad_norm = 0
            for grad in grad_params_coarse:
                grad_norm += grad.pow(2).sum()
            for grad in grad_params_fine:
                grad_norm += grad.pow(2).sum()
            
            grad_norm_loss = grad_norm.sqrt()

            loss += cfg.experiment.grad_norm_rate * grad_norm_loss


            



        optimizer.zero_grad()

        #optimizer_env.zero_grad()
        #loss.backward()
        #loss_off.backward()
        
        loss.backward()
        if cfg.dataset.is_rgb:
            psnr = mse2psnr(fine_loss_off.item())
        else:
            psnr = mse2psnr(fine_loss.item())
        #if no_ir_train == True or jointtrain == True:
        optimizer.step()
        if is_joint and cfg.experiment.grad_norm_rate > 0:
            del grad_params_coarse, grad_params_fine, grad_norm_loss, grad_norm

        #if no_ir_train == False:

        #optimizer_env.step()



        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000

        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
        #print(num_decay_steps, lr_new)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_new

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " PSNR: "
                + str(psnr)
            )
        writer.add_scalar("train/loss", loss.item(), i)
        #writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        writer.add_scalar("train/coarse_loss_off", coarse_loss_off.item(), i)
        writer.add_scalar("train/fine_loss_off", fine_loss_off.item(), i)
        if not cfg.dataset.is_rgb:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
            writer.add_scalar("train/fine_normal_diff_loss", fine_normal_loss.item(), i)
            writer.add_scalar("train/fine_normal_loss_gt", fine_normal_loss_gt.item(), i)
            writer.add_scalar("train/d_normal_loss_gt", d_normal_loss_gt.item(), i)
            writer.add_scalar("train/depth_loss", fine_nerf_depth_loss_gt.item(), i)
            if depth_fine_nerf_backup is not None:
                writer.add_scalar("train/depth_loss_backup", fine_nerf_depth_loss_backup.item(), i)
            writer.add_scalar("train/psnr", psnr, i)


        # Validation
        #if False:
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):

            tqdm.write("[VAL] =======> Iter: " + str(i))
            test_mode = "validation"
            model_coarse.eval()
            model_fine.eval()
            #model_fuse.eval()
            if not cfg.dataset.is_rgb:
                model_env_coarse.eval()
                model_env_fine.eval()

            mean_nerf_abs_err = 0
            mean_nerf_err4 = 0
            mean_nerf_obj_err = 0
            mean_dex_abs_err = 0
            mean_dex_err4 = 0
            mean_dex_obj_err = 0

            start = time.time()
            for img_idx in i_val:
                with torch.no_grad():
                    rgb_coarse, rgb_fine = None, None
                    target_ray_values = None

                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :, :].to(device)
                    ir_extrinsic_target = ir_poses[img_idx,:,:].to(device)
                    depth_target = depths[img_idx].to(device)
                    label_target = labels[img_idx].to(device)
                    img_off_target = imgs_off[img_idx].to(device)
                    normal_target = normals[img_idx].to(device)
                    #print(label_target.shape, label_target[135,240])
                    #assert 1==0
                    intrinsic_target = intrinsics[img_idx,:,:].to(device)
                    ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(
                        H, W, focal, pose_target, intrinsic_target
                    )
                    coords = torch.stack(
                        meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                        dim=-1,
                    )
                    
                    coords = coords.permute(1,0,2)
                    coords = coords.reshape((-1, 2))
                    #print(coords)
                    
                    #assert 1==0
                    #rgb_coarse, _, _, rgb_fine, _, _ ,depth_fine_dex
                    nerf_out = run_one_iter_of_nerf_ir(
                        H,
                        W,
                        intrinsic_target[0,0],
                        model_coarse,
                        model_fine,
                        model_env_coarse,
                        model_env_fine,
                        #model_fuse,
                        ray_origins,
                        ray_directions,
                        cam_origins.cuda(),
                        cam_directions.cuda(),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        m_thres_cand=m_thres_cand,
                        idx = coords,
                        light_extrinsic=ir_extrinsic_target,
                        is_rgb=cfg.dataset.is_rgb,
                        #gt_normal=normal_target.reshape([-1,3])
                    )
                    rgb_coarse, rgb_coarse_off, rgb_fine, rgb_fine_off = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
                    #print(rgb_coarse_off[135,240,:])

                    depth_fine_nerf = nerf_out[8]
                    normal_fine, albedo_fine, roughness_fine = nerf_out[11], nerf_out[12], nerf_out[13]
                    #normals_diff_map = nerf_out[13]
                    depth_fine_dex = list(nerf_out[19:])
                    target_ray_values = img_target.unsqueeze(-1)
                    target_ray_values_off = img_off_target.unsqueeze(-1)

                    coarse_loss = 0.#img2mse(rgb_coarse, target_ray_values)
                    loss, fine_loss = 0.0, 0.0
                    if not cfg.dataset.is_rgb:
                        if rgb_fine is not None:
                            fine_loss = img2mse(rgb_fine, target_ray_values)
                            loss = fine_loss
                        else:
                            loss = coarse_loss
                    else:
                        #print(rgb_fine_off.shape, target_ray_values_off.shape)
                        fine_loss = img2mse(torch.squeeze(rgb_fine_off), torch.squeeze(target_ray_values_off))
                        loss = fine_loss
                       
                    
                    #loss = coarse_loss + fine_loss
                    
                    psnr = mse2psnr(loss.item())
                    writer.add_scalar(test_mode+"/loss", loss.item(), i)
                    #writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                    writer.add_scalar(test_mode+"/psnr", psnr, i)
                    #writer.add_image(
                    #    "validation/rgb_coarse", vutils.make_grid(rgb_coarse[...,0], padding=0, nrow=1, normalize=True, scale_each=True), i
                    #)
                    #print(torch.max(rgb_fine), torch.min(rgb_fine))
                    #assert 1==0

                    gt_depth_torch = depth_target.cpu()
                    img_ground_mask = (gt_depth_torch > 0) & (gt_depth_torch < 1.25)
                    min_err = None
                    min_abs_err = 100000000000.
                    min_abs_depth = None
                    min_cand = 0

                    for cand in range(m_thres_cand.shape[0]):
                    
                        pred_depth_torch = depth_fine_dex[cand].detach().cpu()

                        err = compute_err_metric(gt_depth_torch, pred_depth_torch, img_ground_mask)
                        obj_depth_err_dex, obj_depth_4_err_dex, obj_count_dex = compute_obj_err(gt_depth_torch, pred_depth_torch, label_target.detach().cpu(), img_ground_mask)
                        if obj_depth_err_dex.mean() < min_abs_err:
                            min_abs_err = obj_depth_err_dex.mean()
                            min_err = err
                            min_abs_depth = pred_depth_torch
                            min_cand = m_thres_cand[cand]

                    pred_depth_nerf = depth_fine_nerf.detach().cpu()
                    nerf_err = compute_err_metric(gt_depth_torch, pred_depth_nerf, img_ground_mask)

                    total_obj_depth_err_dex, total_obj_depth_4_err_dex, total_obj_count_dex = compute_obj_err(gt_depth_torch, min_abs_depth, label_target.detach().cpu(), img_ground_mask)
                    total_obj_depth_err_nerf, total_obj_depth_4_err_nerf, total_obj_count_nerf = compute_obj_err(gt_depth_torch, pred_depth_nerf, label_target.detach().cpu(), img_ground_mask)

                    mean_nerf_abs_err += nerf_err['depth_abs_err']
                    mean_nerf_err4 += nerf_err['depth_err4']
                    mean_nerf_obj_err += total_obj_depth_err_nerf

                    mean_dex_abs_err += min_err['depth_abs_err']
                    mean_dex_err4 += min_err['depth_err4']
                    mean_dex_obj_err += total_obj_depth_err_dex

                    rgb_fine_np = rgb_fine.cpu().numpy()[:,:,0]
                    img_target_np = img_target.cpu().numpy()

                    rgb_fine_np = (rgb_fine_np*255).astype(np.uint8)
                    img_target_np = (img_target_np*255).astype(np.uint8)

                    rgb_fine_np_img = Image.fromarray(rgb_fine_np, mode='L')
                    img_target_np_img = Image.fromarray(img_target_np, mode='L')
                    if debug_output:
                        rgb_fine_np_img.save(os.path.join(logdir,"pred_nerf",test_mode+"_pred_nerf_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                        img_target_np_img.save(os.path.join(logdir,"pred_nerf_gt",test_mode+"_pred_nerf_gt_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                        #print(np.max(rgb_fine_np), np.min(rgb_fine_np), np.max(img_target_np), np.min(img_target_np))
                        
                        #assert 1==0
                    
                        pred_depth_nerf_np = pred_depth_nerf.numpy()
                        depth_pts = depth2pts_np(pred_depth_nerf_np, intrinsic_target.cpu().numpy(), pose_target.cpu().numpy())
                        pts_o3d = o3d.utility.Vector3dVector(depth_pts)
                        pcd = o3d.geometry.PointCloud(pts_o3d)
                        o3d.io.write_point_cloud(os.path.join(logdir,"pred_depth_pcd_nerf",test_mode+"_pred_depth_pcd_step_"+str(i)+ "_" + str(img_idx) + ".ply"), pcd)

                        depth_np_gt = depth_target.cpu().numpy()
                        depth_pts = depth2pts_np(depth_np_gt, intrinsic_target.cpu().numpy(), pose_target.cpu().numpy())
                        pts_o3d = o3d.utility.Vector3dVector(depth_pts)
                        pcd = o3d.geometry.PointCloud(pts_o3d)
                        o3d.io.write_point_cloud(os.path.join(logdir,"pred_depth_pcd_nerf_gt",test_mode+"_gt_depth_pcd_step_"+str(i)+ "_" + str(img_idx) + ".ply"), pcd)


                    #print(pred_depth_nerf.shape, depth_target.shape)
                    #assert 1==0


                    pred_depth_nerf_np = pred_depth_nerf_np*1000
                    pred_depth_nerf_np = (pred_depth_nerf_np).astype(np.uint32)
                    out_pred_depth_nerf = Image.fromarray(pred_depth_nerf_np, mode='I')
                    out_pred_depth_nerf.save(os.path.join(logdir,"pred_depth_nerf",test_mode+"_pred_depth_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                    pred_depth_nerf_err_np = depth_error_img((pred_depth_nerf.unsqueeze(0))*1000, (gt_depth_torch.unsqueeze(0))*1000, img_ground_mask.unsqueeze(0))
                    pred_depth_nerf_err_np_img = (pred_depth_nerf_err_np*255).astype(np.uint8)
                    pred_depth_nerf_err_np_img = Image.fromarray(pred_depth_nerf_err_np_img, mode='RGB')
                    pred_depth_nerf_err_np_img.save(os.path.join(logdir,"pred_depth_err_nerf",test_mode+"_pred_depth_err_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                    with open(os.path.join(logdir,"pred_depth_err_nerf", test_mode+"_output_result.txt"), "a") as f:
                        f.write("iter: "
                        + str(i)
                        + " img_idx: "
                        + str(img_idx)
                        + " Nerf Abs Err: "
                        + str(nerf_err['depth_abs_err'])
                        + " Nerf Err4: "
                        + str(nerf_err['depth_err4'])
                        + " Nerf Obj Err: "
                        + str(total_obj_depth_err_nerf)
                        + "\n"
                        )

                    pred_depth_np = min_abs_depth.numpy()
                    pred_depth_np = pred_depth_np*1000
                    pred_depth_np = (pred_depth_np).astype(np.uint32)
                    out_pred_depth = Image.fromarray(pred_depth_np, mode='I')
                    out_pred_depth.save(os.path.join(logdir,"pred_depth_dex",test_mode+"_pred_depth_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                    pred_depth_err_np = depth_error_img((min_abs_depth.unsqueeze(0))*1000, (gt_depth_torch.unsqueeze(0))*1000, img_ground_mask.unsqueeze(0))
                    pred_depth_err_np_img = (pred_depth_err_np*255).astype(np.uint8)
                    pred_depth_err_np_img = Image.fromarray(pred_depth_err_np_img, mode='RGB')
                    pred_depth_err_np_img.save(os.path.join(logdir,"pred_depth_err_dex",test_mode+"_pred_depth_err_step_"+str(i)+ "_" + str(img_idx) + ".png"))
                    with open(os.path.join(logdir,"pred_depth_err_dex", test_mode+"_output_result.txt"), "a") as f:
                        f.write("iter: "
                        + str(i)
                        + " img_idx: "
                        + str(img_idx)
                        + " Dex Abs Err: "
                        + str(min_err['depth_abs_err'])
                        + " Dex Err4: "
                        + str(min_err['depth_err4'])
                        + " Dex Obj Err: "
                        + str(total_obj_depth_err_dex)
                        + "\n"
                        )


            mean_nerf_abs_err = mean_nerf_abs_err/len(i_val)
            mean_nerf_err4 = mean_nerf_err4/len(i_val)
            mean_nerf_obj_err = mean_nerf_obj_err/len(i_val)

            mean_dex_abs_err = mean_dex_abs_err/len(i_val)
            mean_dex_err4 = mean_dex_err4/len(i_val)
            mean_dex_obj_err = mean_dex_obj_err/len(i_val)

                    
            if rgb_fine_off is not None:
                if not cfg.dataset.is_rgb:
                    normal_fine = normal_fine.permute(2,0,1)
                    normal_fine = (normal_fine.clone().detach()*0.5+0.5)
                    normal_target = normal_target.permute(2,0,1)
                    normal_target = (normal_target.clone().detach()*0.5+0.5)
                    
                    #print(torch.max(albedo_fine), torch.min(albedo_fine), torch.max(roughness_fine), torch.min(roughness_fine))
                    writer.add_image(
                        test_mode+"/normal_fine", vutils.make_grid(normal_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/albedo_fine", vutils.make_grid(albedo_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/roughness_fine", vutils.make_grid(roughness_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/normal_gt", vutils.make_grid(normal_target, padding=0, nrow=1), i
                    )
                    ir_light = model_env_fine.ir_pattern.clone().detach()
                    ir_light_out = torch.nn.functional.softplus(ir_light, beta=5)
                    #print(ir_light.shape)
                    #assert 1==0
                    writer.add_image(
                        test_mode+"/ir_light", vutils.make_grid(ir_light_out, padding=0, nrow=1, normalize=True), i
                    )
                    
                    writer.add_image(
                        test_mode+"/rgb_fine", vutils.make_grid(rgb_fine[...,0], padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_fine_off", vutils.make_grid(rgb_fine_off[...,0], padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_coarse_off", vutils.make_grid(rgb_coarse_off[...,0], padding=0, nrow=1), i
                    )
                else:
                    writer.add_image(
                        test_mode+"/rgb_fine_off", vutils.make_grid(rgb_fine_off[...,:].permute(2,0,1), padding=0, nrow=1), i
                    )
                    writer.add_image(
                        test_mode+"/rgb_coarse_off", vutils.make_grid(rgb_coarse_off[...,:].permute(2,0,1), padding=0, nrow=1), i
                    )
                writer.add_scalar(test_mode+"/fine_loss", fine_loss.item(), i)
            #print(cast_to_image(target_ray_values[..., :3]).shape, type(cast_to_image(target_ray_values[..., :3])))
            writer.add_image(
                test_mode+"/img_target",
                vutils.make_grid(target_ray_values[...,0], padding=0, nrow=1),
                i,
            )
            if not cfg.dataset.is_rgb:
                writer.add_image(
                    test_mode+"/img_off_target",
                    vutils.make_grid(img_off_target, padding=0, nrow=1),
                    i,
                )
            else:
                writer.add_image(
                    test_mode+"/img_off_target",
                    vutils.make_grid(img_off_target.permute(2,0,1), padding=0, nrow=1),
                    i,
                )
            
            #torch.save(model_env_fine.ir_pattern, os.path.join(logdir,'ir_pat.pt'))

            #print((torch.sum((target_ray_values[...,0]-img_off_target)<0)))
            #assert 1==0

            

            #print(rgb_fine[...,0].shape, target_ray_values[...,0].shape, rgb_fine_off[...,0].shape, img_off_target.shape,img_ground_mask.unsqueeze(0).shape)

            if not cfg.dataset.is_rgb:
                pred_rgb_fine_err_np = render_error_img(rgb_fine[...,0], target_ray_values[...,0], img_ground_mask)
                pred_rgb_fine_off_err_np = render_error_img(rgb_fine_off[...,0], img_off_target, img_ground_mask)


                writer.add_image(
                    test_mode+"/rgb_fine_err",
                    pred_rgb_fine_err_np.transpose((2,0,1)),
                    i,
                )
                writer.add_image(
                    test_mode+"/rgb_fine_off_err",
                    pred_rgb_fine_off_err_np.transpose((2,0,1)),
                    i,
                )

            
            #print(rgb_fine[...,0].shape, torch.max(rgb_fine[...,0]), torch.min(rgb_fine[...,0]))
            #assert 1==0
            writer.add_image(
                    test_mode+"/depth_pred_nerf_err",
                    pred_depth_nerf_err_np.transpose((2,0,1)),
                    i,
                )
            writer.add_image(
                    test_mode+"/depth_pred_nerf",
                    vutils.make_grid(pred_depth_nerf, padding=0, nrow=1, normalize=True, scale_each=True),
                    i,
                )

            
            #print(type(min_abs_depth), min_abs_depth.shape)
            writer.add_image(
                    test_mode+"/depth_pred_dex",
                    vutils.make_grid(min_abs_depth, padding=0, nrow=1, normalize=True, scale_each=True),
                    i,
                )
            

            
            #print(total_obj_depth_err, total_obj_depth_4_err, total_obj_count)
            #assert 1==0
                
            
            #print(pred_depth_err_np.transpose((1,2,0)).shape)
            writer.add_image(
                    test_mode+"/depth_pred_err",
                    pred_depth_err_np.transpose((2,0,1)),
                    i,
                )


                #print(depth_fine_dex[cand].shape)
            writer.add_image(
                test_mode+"/depth_gt",
                vutils.make_grid(depth_target, padding=0, nrow=1, normalize=True, scale_each=True),
                i,
            )
            tqdm.write(
                "Validation loss: "
                + str(loss.item())
                + " Validation PSNR: "
                + str(psnr)
                + " Time: "
                + str(time.time() - start)
                + " Dex Abs Err: "
                + str(mean_dex_abs_err)
                + " Dex Err4: "
                + str(mean_dex_err4)
                + " Nerf Abs Err: "
                + str(mean_nerf_abs_err)
                + " Nerf Err4: "
                + str(mean_nerf_err4)
                + " Dex Obj Err: "
                + str(mean_dex_obj_err)
                + " Nerf Obj Err: "
                + str(mean_nerf_obj_err)
                + " Best Thres: "
                + str(min_cand)
            )
            with open(os.path.join(logdir, test_mode+"_output_result.txt"), "a") as f:
                f.write("iter: "
                + str(i)
                + " Validation loss: "
                + str(loss.item())
                + " Validation PSNR: "
                + str(psnr)
                + " Time: "
                + str(time.time() - start)
                + " Dex Abs Err: "
                + str(min_err['depth_abs_err'])
                + " Dex Err4: "
                + str(min_err['depth_err4'])
                + " Nerf Abs Err: "
                + str(nerf_err['depth_abs_err'])
                + " Nerf Err4: "
                + str(nerf_err['depth_err4'])
                + " Dex Obj Err: "
                + str(total_obj_depth_err_dex)
                + " Nerf Obj Err: "
                + str(total_obj_depth_err_nerf)
                + "\n"
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            if not cfg.dataset.is_rgb:
                checkpoint_dict = {
                    "iter": i,
                    "model_coarse_state_dict": model_coarse.state_dict(),
                    "model_fine_state_dict": model_fine.state_dict(),
                    "model_env_coarse_state_dict": model_env_coarse.state_dict(),
                    "model_env_fine_state_dict": model_env_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr
                }
            else:
                checkpoint_dict = {
                    "iter": i,
                    "model_coarse_state_dict": model_coarse.state_dict(),
                    "model_fine_state_dict": model_fine.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                    "psnr": psnr
                }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor, color_channel=3):
    #print(tensor.shape)
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    #print(img.shape)
    return img

def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic=np.eye(4)):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid

if __name__ == "__main__":
    main()
