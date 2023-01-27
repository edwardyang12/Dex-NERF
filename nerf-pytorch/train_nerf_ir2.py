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

from nerf import compute_err_metric, depth_error_img, compute_obj_err

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_blender_data, load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, load_messytable_data,run_one_iter_of_nerf_ir)


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
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf = None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            images, poses, render_poses, hwf, i_split, intrinsics, depths, labels, imgs_off, normals = load_messytable_data(
                cfg.dataset.basedir,
                half_res=cfg.dataset.half_res,
                debug = False,
                testskip=cfg.dataset.testskip,
                cfg=cfg,
                is_real_rgb=cfg.dataset.is_real_rgb,
                sceneid = configargs.sceneid
            )
            #print(images.shape, i_split)
            #assert 1==0
            i_train, i_val, i_test = i_split
            H, W, _ = hwf
            H, W = int(H), int(W)
            if cfg.nerf.train.white_background:
                images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        elif cfg.dataset.type.lower() == "llff":
            images, poses, bds, render_poses, i_test = load_llff_data(
                cfg.dataset.basedir, factor=cfg.dataset.downsample_factor
            )
            hwf = poses[0, :3, -1]
            poses = poses[:, :, :]
            if not isinstance(i_test, list):
                i_test = [i_test]
            if cfg.dataset.llffhold > 0:
                i_test = np.arange(images.shape[0])[:: cfg.dataset.llffhold]
            i_val = i_test
            i_train = np.array(
                [
                    i
                    for i in np.arange(images.shape[0])
                    if (i not in i_test and i not in i_val)
                ]
            )
            H, W, focal = hwf
            H, W = int(H), int(W)
            hwf = [H, W, focal]
            images = torch.from_numpy(images)
            poses = torch.from_numpy(poses)

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
        color_channel=1
    )
    model_coarse.to(device)
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
        W = cfg.dataset.W
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
        W = cfg.dataset.W
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
            color_channel=1
        )
        model_fine.to(device)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    trainable_parameters_env = list(model_env_coarse.parameters())
    #trainable_parameters_env += list(model_fuse.parameters())
    trainable_parameters += list(model_fine.parameters())
    trainable_parameters_env += list(model_env_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
    )
    optimizer_env = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters_env, lr=cfg.optimizer.lr
    )

    
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, str(configargs.sceneid))
    os.makedirs(logdir, exist_ok=True)
    
    m_thres_max = cfg.nerf.validation.m_thres
    m_thres_cand = np.arange(5,m_thres_max+5,5)
    os.makedirs(os.path.join(logdir,"pred_depth_dex"), exist_ok=True)
    os.makedirs(os.path.join(logdir,"pred_depth_nerf"), exist_ok=True)

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
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays
    #no_ir_train = True
    #jointtrain = False
    is_joint = False

    #prev_params = list(model_env_fine.parameters())

    for i in trange(start_iter, cfg.experiment.train_iters):

        model_coarse.train()
        model_env_coarse.train()
        model_fine.train()
        model_env_fine.train()


        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        if USE_CACHED_DATASET:
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            #rgb_coarse, _, _, rgb_fine, _, _,_
            nerf_out = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                m_thres_cand=m_thres_cand
            )
            rgb_coarse, rgb_fine = nerf_out[0], nerf_out[3]
        else:
            img_idx = np.random.choice(i_train)
            img_target = images[img_idx].to(device)
            
            pose_target = poses[img_idx, :, :].to(device)
            depth_target = depths[img_idx].to(device)
            normal_target = normals[img_idx].to(device)
            #print(normal_target.shape)
            #assert 1==0
            #print(img_target.shape, depth_target.shape)
            #assert 1==0
            #print("===========================================")
            #print(pose_target)

            #print(pose_target.shape)
            intrinsic_target = intrinsics[img_idx,:,:].to(device)
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
            

            #print(target_s.shape)
            #assert 1==0
            then = time.time()
            #print(ray_origins.shape, ray_directions.shape)
            #print(ray_origins[-3:,:], ray_directions[-3:,:])
            #rgb_coarse, _, _, rgb_fine, _, _, _
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
                joint=is_joint
            )
            rgb_coarse, rgb_off_coarse, rgb_fine, rgb_off_fine = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
            alpha_fine = nerf_out[9]
            normal_fine = nerf_out[10]
            normals_diff_map = nerf_out[13]
            d_n_map = nerf_out[14]
            albedo_cost_map = nerf_out[15]
            roughness_cost_map = nerf_out[16]

            #rgb_coarse = torch.mean(rgb_coarse, dim=-1)
            #rgb_fine = torch.mean(rgb_fine, dim=-1)
            #print(rgb_coarse.shape, rgb_fine.shape)
            target_ray_values = target_s.unsqueeze(-1)
            target_ray_values_off = target_s_off.unsqueeze(-1)
            #print(rgb_coarse.shape, rgb_fine.shape, target_ray_values.shape)
            #assert 1==0
        
        #if i == cfg.experiment.finetune_start:
        #    no_ir_train = False
        if i == cfg.experiment.jointtrain_start:
            is_joint = True
        coarse_loss = 0.0
        #print(torch.max(model_env_fine.ir_pattern))

        #print(normal_fine.shape, target_n.shape)
        #assert 1==0
        """
        if no_ir_train == False:
            coarse_loss = torch.nn.functional.mse_loss(
                rgb_coarse, target_ray_values
            )

        if no_ir_train == True or jointtrain == True:
            coarse_loss += torch.nn.functional.mse_loss(
                rgb_off_coarse, target_ray_values_off
            )
        #print(rgb_off_coarse.shape, target_ray_values_off.shape)
        #assert 1==0
       
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = 0.0

            if no_ir_train == False:
                fine_loss = torch.nn.functional.mse_loss(
                        rgb_fine, target_ray_values
                )

            if no_ir_train == True or jointtrain == True:
                fine_loss += torch.nn.functional.mse_loss(
                    rgb_off_fine, target_ray_values_off
                )
            #fine_loss = fine_loss + fine_loss_off


        if configargs.depth_supervise == True:
            print(target_d.shape, alpha_fine.shape)
            #depth_loss = img2mse(depth_fine_dex[0], depth_target)
            #print(depth_fine_dex[0].shape, depth_loss)
            assert 1==0
            
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        
        #if no_ir_train == True or jointtrain == True:
        optimizer.zero_grad()
        #if no_ir_train == False:
        optimizer_env.zero_grad()
        loss.backward()
        psnr = mse2psnr(loss.item())
        #if no_ir_train == True or jointtrain == True:
        optimizer.step()
        #if no_ir_train == False:
        optimizer_env.step()
        """

        #coarse_loss = torch.nn.functional.mse_loss(
        #        rgb_coarse, target_ray_values
        #)

        
        coarse_loss_off = torch.nn.functional.mse_loss(
            rgb_off_coarse, target_ray_values_off
        )
        fine_loss = torch.nn.functional.mse_loss(
                rgb_fine, target_ray_values
        )
        fine_loss_off = torch.nn.functional.mse_loss(
            rgb_off_fine, target_ray_values_off
        )
        d_normal_loss_gt = torch.nn.functional.mse_loss(
            d_n_map, target_n
        )
        fine_normal_loss_gt = torch.nn.functional.mse_loss(
            normal_fine, target_n
        )
        #print(d_n_map.shape, (target_n*(-1.)).shape)
        #assert 1==0
        fine_normal_loss = normals_diff_map.mean()

        albedo_smoothness_loss = torch.mean(albedo_cost_map)
        roughness_smoothness_loss = torch.mean(roughness_cost_map)
        #print(fine_normal_loss)

        #if i < 10000:
        #    fine_normal_loss = fine_normal_loss*0.

        loss_off = coarse_loss_off + fine_loss_off
        #print(fine_loss.item(), fine_normal_loss.item())
        loss_on = fine_loss + \
               cfg.experiment.normal_gt_rate * fine_normal_loss_gt + \
               cfg.experiment.normal_derived_rate * fine_normal_loss + \
               cfg.experiment.albedo_rate * albedo_smoothness_loss + \
               cfg.experiment.roughness_rate * roughness_smoothness_loss
        loss = cfg.experiment.ir_on_rate * loss_on + cfg.experiment.ir_off_rate * loss_off

        optimizer.zero_grad()

        optimizer_env.zero_grad()
        #loss.backward()
        #loss_off.backward()
        loss.backward()
        psnr = mse2psnr(fine_loss.item())
        #if no_ir_train == True or jointtrain == True:
        optimizer.step()
        #if no_ir_train == False:

        optimizer_env.step()

        #issame = True
        #for params in zip(prev_params, list(model_env_fine.parameters())):
        #    p1,p2 = params
        #    if not torch.all(p1 == p2):
        #        issame = False
        #        break

        #print(p1,p2)
        #print(issame)
        #print(model_env_fine.attenuation)
        #prev_params = copy.deepcopy(list(model_env_fine.parameters()))
        #assert 1==0
        

        # Learning rate updates
        num_decay_steps = cfg.scheduler.lr_decay * 1000
        lr_new = cfg.optimizer.lr * (
            cfg.scheduler.lr_decay_factor ** (i / num_decay_steps)
        )
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
        writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/fine_loss_off", fine_loss_off.item(), i)
        writer.add_scalar("train/fine_normal_diff_loss", fine_normal_loss.item(), i)
        writer.add_scalar("train/fine_normal_loss_gt", fine_normal_loss_gt.item(), i)
        writer.add_scalar("train/d_normal_loss_gt", d_normal_loss_gt.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        #print(torch.max(model_env_fine.ir_pattern))

        #writer.add_image(
        #            "train/img_target",
        #            cast_to_image(img_target[..., :3]),
        #            i,
        #        )
        #assert 1==0
        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            model_env_coarse.eval()
            #model_fuse.eval()
            model_fine.eval()
            model_env_fine.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    #rgb_coarse, _, _, rgb_fine, _, _ ,_
                    nerf_out = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        m_thres_cand=m_thres_cand
                    )
                    rgb_coarse, rgb_fine = nerf_out[0], nerf_out[3]
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)
                    pose_target = poses[img_idx, :, :].to(device)
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
                        idx = coords
                    )
                    rgb_coarse, rgb_coarse_off, rgb_fine, rgb_fine_off = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
                    depth_fine_nerf = nerf_out[8]
                    normal_fine, albedo_fine, roughness_fine = nerf_out[10], nerf_out[11], nerf_out[12]
                    #normals_diff_map = nerf_out[13]
                    depth_fine_dex = list(nerf_out[17:])
                    target_ray_values = img_target.unsqueeze(-1)
                    #print(rgb_coarse.shape,rgb_fine.shape, target_ray_values.shape)
                    #rgb_coarse = torch.mean(rgb_coarse, dim=-1)
                    #rgb_fine = torch.mean(rgb_fine, dim=-1)
                #print(target_ray_values.shape, rgb_coarse.shape)
                #assert 1==0
                #print(depth_fine_dex.shape)
                #print(rgb_coarse.shape, target_ray_values.shape)
                coarse_loss = 0.#img2mse(rgb_coarse, target_ray_values)
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine, target_ray_values)
                    loss = fine_loss
                else:
                    loss = coarse_loss
                
                loss = coarse_loss + fine_loss
                
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                #writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                #writer.add_image(
                #    "validation/rgb_coarse", vutils.make_grid(rgb_coarse[...,0], padding=0, nrow=1, normalize=True, scale_each=True), i
                #)
                #print(torch.max(rgb_fine), torch.min(rgb_fine))
                #assert 1==0
                writer.add_image(
                    "validation/rgb_coarse_off", vutils.make_grid(rgb_coarse_off[...,0], padding=0, nrow=1), i
                )
                if rgb_fine is not None:
                    normal_fine = normal_fine.permute(2,0,1)
                    normal_fine = (normal_fine.clone().detach()*0.5+0.5)
                    normal_target = normal_target.permute(2,0,1)
                    normal_target = (normal_target.clone().detach()*0.5+0.5)
                    
                    #print(torch.max(albedo_fine), torch.min(albedo_fine), torch.max(roughness_fine), torch.min(roughness_fine))
                    writer.add_image(
                        "validation/normal_fine", vutils.make_grid(normal_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        "validation/albedo_fine", vutils.make_grid(albedo_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        "validation/roughness_fine", vutils.make_grid(roughness_fine, padding=0, nrow=1), i
                    )
                    writer.add_image(
                        "validation/normal_gt", vutils.make_grid(normal_target, padding=0, nrow=1), i
                    )
                    ir_light = model_env_fine.ir_pattern.clone().detach()
                    ir_light_out = torch.nn.functional.softplus(ir_light, beta=5)
                    #print(ir_light.shape)
                    #assert 1==0
                    writer.add_image(
                        "validation/ir_light", vutils.make_grid(ir_light_out, padding=0, nrow=1, normalize=True), i
                    )
                    
                    writer.add_image(
                        "validation/rgb_fine", vutils.make_grid(rgb_fine[...,0], padding=0, nrow=1), i
                    )
                    writer.add_image(
                        "validation/rgb_fine_off", vutils.make_grid(rgb_fine_off[...,0], padding=0, nrow=1), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                #print(cast_to_image(target_ray_values[..., :3]).shape, type(cast_to_image(target_ray_values[..., :3])))
                writer.add_image(
                    "validation/img_target",
                    vutils.make_grid(target_ray_values[...,0], padding=0, nrow=1),
                    i,
                )
                writer.add_image(
                    "validation/img_off_target",
                    vutils.make_grid(img_off_target, padding=0, nrow=1),
                    i,
                )
                #print((torch.sum((target_ray_values[...,0]-img_off_target)<0)))
                #assert 1==0

                gt_depth_torch = depth_target.cpu()
                img_ground_mask = (gt_depth_torch > 0) & (gt_depth_torch < 1.25)
                min_err = None
                min_abs_err = 100000000000.
                min_abs_depth = None
                min_cand = 0

                pred_depth_nerf = depth_fine_nerf.detach().cpu()
                nerf_err = compute_err_metric(gt_depth_torch, pred_depth_nerf, img_ground_mask)
                pred_depth_nerf_np = pred_depth_nerf.numpy()
                pred_depth_nerf_np = pred_depth_nerf_np*1000
                pred_depth_nerf_np = (pred_depth_nerf_np).astype(np.uint32)
                out_pred_depth_nerf = Image.fromarray(pred_depth_nerf_np, mode='I')
                out_pred_depth_nerf.save(os.path.join(logdir,"pred_depth_nerf","pred_depth_step_"+str(i)+".png"))
                pred_depth_nerf_err_np = depth_error_img((pred_depth_nerf.unsqueeze(0))*1000, (gt_depth_torch.unsqueeze(0))*1000, img_ground_mask.unsqueeze(0))
                writer.add_image(
                        "validation/depth_pred_nerf_err",
                        pred_depth_nerf_err_np.transpose((2,0,1)),
                        i,
                    )
                writer.add_image(
                        "validation/depth_pred_nerf",
                        vutils.make_grid(pred_depth_nerf, padding=0, nrow=1, normalize=True, scale_each=True),
                        i,
                    )

                for cand in range(m_thres_cand.shape[0]):
                    #print(m_thres_cand[cand], obj_depth_err_dex.mean())

                
                    pred_depth_torch = depth_fine_dex[cand].detach().cpu()
                    
                    #print(gt_depth_torch.shape, pred_depth_torch.shape, img_ground_mask.shape)
                    #assert 1==0
                    err = compute_err_metric(gt_depth_torch, pred_depth_torch, img_ground_mask)
                    obj_depth_err_dex, obj_depth_4_err_dex, obj_count_dex = compute_obj_err(gt_depth_torch, pred_depth_torch, label_target.detach().cpu(), img_ground_mask)
                    if obj_depth_err_dex.mean() < min_abs_err:
                        min_abs_err = obj_depth_err_dex.mean()
                        min_err = err
                        min_abs_depth = pred_depth_torch
                        min_cand = m_thres_cand[cand]
                #print(type(min_abs_depth), min_abs_depth.shape)
                writer.add_image(
                        "validation/depth_pred_dex",
                        vutils.make_grid(min_abs_depth, padding=0, nrow=1, normalize=True, scale_each=True),
                        i,
                    )
                

                total_obj_depth_err_dex, total_obj_depth_4_err_dex, total_obj_count_dex = compute_obj_err(gt_depth_torch, min_abs_depth, label_target.detach().cpu(), img_ground_mask)
                total_obj_depth_err_nerf, total_obj_depth_4_err_nerf, total_obj_count_nerf = compute_obj_err(gt_depth_torch, pred_depth_nerf, label_target.detach().cpu(), img_ground_mask)
                #print(total_obj_depth_err, total_obj_depth_4_err, total_obj_count)
                #assert 1==0
                    
                pred_depth_np = min_abs_depth.numpy()
                pred_depth_np = pred_depth_np*1000
                pred_depth_np = (pred_depth_np).astype(np.uint32)
                out_pred_depth = Image.fromarray(pred_depth_np, mode='I')
                out_pred_depth.save(os.path.join(logdir,"pred_depth_dex","pred_depth_step_"+str(i)+".png"))

                pred_depth_err_np = depth_error_img((min_abs_depth.unsqueeze(0))*1000, (gt_depth_torch.unsqueeze(0))*1000, img_ground_mask.unsqueeze(0))
                #print(pred_depth_err_np.transpose((1,2,0)).shape)
                writer.add_image(
                        "validation/depth_pred_err",
                        pred_depth_err_np.transpose((2,0,1)),
                        i,
                    )


                    #print(depth_fine_dex[cand].shape)
                writer.add_image(
                    "validation/depth_gt",
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
                    + str(min_abs_err)
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
                    + " Best Thres: "
                    + str(min_cand)
                )

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "model_env_coarse_state_dict": model_env_coarse.state_dict(),
                "model_env_fine_state_dict": None
                if not model_env_fine
                else model_env_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
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


if __name__ == "__main__":
    main()
