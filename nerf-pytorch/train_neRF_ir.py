import argparse
import glob
import os
import time

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
                  mse2psnr, run_one_iter_of_nerf, load_messytable_data_RF,run_one_iter_of_nerf_ir,
                  SgRenderer, run_one_iter_of_neRF)


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
        default=1,
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

    images, poses, render_poses, hwf, i_split, intrinsics, depths, labels, imgs_off, sid = load_messytable_data_RF(
        cfg.dataset.basedir,
        half_res=cfg.dataset.half_res,
        debug = False,
        testskip=cfg.dataset.testskip,
        cfg=cfg,
        is_real_rgb=cfg.dataset.is_real_rgb,
        sceneid = configargs.sceneid
    )
    #print(images.shape, sid.shape, sid)
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
        include_input_xyz=cfg.models.coarse.include_input_xyz,
        color_channel=1
    )
    model_coarse.to(device)

    model_fine = getattr(models, cfg.models.fine.type)(
        num_layers=cfg.models.fine.num_layers,
        hidden_size=cfg.models.fine.hidden_size,
        skip_connect_every=cfg.models.fine.skip_connect_every,
        num_encoding_fn_xyz=cfg.models.fine.num_encoding_fn_xyz,
        include_input_xyz=cfg.models.fine.include_input_xyz,
        color_channel=1
    )
    model_fine.to(device)

    model_env = getattr(models, cfg.models.env.type)(
        num_scenes=21,
        num_lobes=24
    )
    model_env.to(device)

    SGrender = SgRenderer(True, True, True)

    # Initialize optimizer.
    trainable_parameters = list(model_coarse.parameters())
    #trainable_parameters_env = list(model_env_coarse.parameters())
    #trainable_parameters_env += list(model_fuse.parameters())
    trainable_parameters += list(model_fine.parameters())
    trainable_parameters += list(model_env.parameters())
    #trainable_parameters_env += list(model_env_fine.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(
        trainable_parameters, lr=cfg.optimizer.lr
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
    no_ir_train = True
    jointtrain = False
    for i in trange(start_iter, cfg.experiment.train_iters):
        print(i)
        model_coarse.train()
        model_fine.train()
        model_env.train()


        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        
        img_idx = np.random.choice(i_train)
        img_target = images[img_idx].to(device)
        
        pose_target = poses[img_idx, :, :].to(device)
        depth_target = depths[img_idx].to(device)
        sceneid = sid[img_idx]


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
        #print(target_s.shape)
        #assert 1==0
        #target_env = pattern_target[select_inds[:, 0], select_inds[:, 1]]
        target_d = depth_target[select_inds[:, 0], select_inds[:, 1]]
        target_s_off = img_off_target[select_inds[:, 0], select_inds[:, 1]]
        

        #print(target_s.shape)
        #assert 1==0
        then = time.time()
        #print(ray_origins.shape, ray_directions.shape)
        #print(ray_origins[-3:,:], ray_directions[-3:,:])
        #rgb_coarse, _, _, rgb_fine, _, _, _
        nerf_out = run_one_iter_of_neRF(
            H,
            W,
            intrinsic_target[0,0],
            sceneid,
            model_coarse,
            model_fine,
            model_env,
            SGrender,
            #model_fuse,
            ray_origins,
            ray_directions,
            cam_origins.cuda(),
            cam_directions.cuda(),
            cfg,
            mode="train",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand
        )
        rgb_coarse, rgb_fine = nerf_out[0], nerf_out[3]
        #alpha_fine = nerf_out[9]

        #rgb_coarse = torch.mean(rgb_coarse, dim=-1)
        #rgb_fine = torch.mean(rgb_fine, dim=-1)
        #print(rgb_coarse.shape, rgb_fine.shape)
        target_ray_values = target_s.unsqueeze(-1)
        target_ray_values_off = target_s_off.unsqueeze(-1)
        #print(rgb_coarse.shape, rgb_fine.shape, target_ray_values.shape)
        #assert 1==0
        

        coarse_loss = 0.0

        coarse_loss += torch.nn.functional.mse_loss(
            rgb_coarse, target_ray_values_off
        )
        #print(rgb_off_coarse.shape, target_ray_values_off.shape)
        #assert 1==0
       
        fine_loss = 0.0
        fine_loss += torch.nn.functional.mse_loss(
            rgb_fine, target_ray_values_off
        )
        #fine_loss = fine_loss + fine_loss_off


            
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss
        loss = coarse_loss + fine_loss
        

        optimizer.zero_grad()
        loss.backward()
        psnr = mse2psnr(loss.item())
        optimizer.step()

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
        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        #writer.add_scalar("train/coarse_loss_off", coarse_loss_off.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        #writer.add_image(
        #            "train/img_target",
        #            cast_to_image(img_target[..., :3]),
        #            i,
        #        )
        #assert 1==0
        # Validation
        if ( False
            #i % cfg.experiment.validate_every == 0
            #or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            #model_fuse.eval()
            model_fine.eval()
            model_env.eval()

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
                    sceneid = sid[img_idx]
                    #print(label_target.shape, label_target[135,240])
                    #assert 1==0
                    intrinsic_target = intrinsics[img_idx,:,:].to(device)
                    ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(
                        H, W, focal, pose_target, intrinsic_target
                    )
                    #rgb_coarse, _, _, rgb_fine, _, _ ,depth_fine_dex
                    nerf_out = run_one_iter_of_neRF(
                        H,
                        W,
                        intrinsic_target[0,0],
                        sceneid,
                        model_coarse,
                        model_fine,
                        model_env,
                        SGrender,
                        #model_fuse,
                        ray_origins,
                        ray_directions,
                        cam_origins.cuda(),
                        cam_directions.cuda(),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        m_thres_cand=m_thres_cand
                    )
                    rgb_coarse, rgb_fine = nerf_out[0], nerf_out[3]
                    depth_fine_nerf = nerf_out[6]
                    depth_fine_dex = list(nerf_out[7:])
                    target_ray_values = img_target.unsqueeze(-1)
                    #print(rgb_coarse.shape,rgb_fine.shape, target_ray_values.shape)
                    #rgb_coarse = torch.mean(rgb_coarse, dim=-1)
                    #rgb_fine = torch.mean(rgb_fine, dim=-1)
                #print(target_ray_values.shape, rgb_coarse.shape)
                #assert 1==0
                #print(depth_fine_dex.shape)
                #print(rgb_coarse.shape, target_ray_values.shape)
                coarse_loss = img2mse(rgb_coarse, target_ray_values)
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine, target_ray_values)
                    loss = fine_loss
                else:
                    loss = coarse_loss
                
                loss = coarse_loss + fine_loss
                
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", vutils.make_grid(rgb_coarse[...,0], padding=0, nrow=1, normalize=True, scale_each=True), i
                )

                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", vutils.make_grid(rgb_fine[...,0], padding=0, nrow=1, normalize=True, scale_each=True), i
                    )

                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
                #print(cast_to_image(target_ray_values[..., :3]).shape, type(cast_to_image(target_ray_values[..., :3])))
                writer.add_image(
                    "validation/img_target",
                    vutils.make_grid(target_ray_values[...,0], padding=0, nrow=1, normalize=True, scale_each=True),
                    i,
                )
                writer.add_image(
                    "validation/img_off_target",
                    vutils.make_grid(img_off_target, padding=0, nrow=1, normalize=True, scale_each=True),
                    i,
                )

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
                "model_env_coarse_state_dict": model_env.state_dict(),
                "model_env_fine_state_dict": None
                if not model_env
                else model_env.state_dict(),
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
