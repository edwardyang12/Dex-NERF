import argparse
import glob
import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
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

    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split = None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None

    # Load dataset
    images, poses, render_poses, hwf = None, None, None, None

    images, poses, ir_poses, render_poses, hwf, i_split, intrinsics, depths, labels, imgs_off, normals = load_messytable_data(
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
        ir_extrinsic=ir_extrinsic
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
        ir_extrinsic=ir_extrinsic
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
    #trainable_parameters_env = list(model_env_coarse.parameters())
    trainable_parameters += list(model_env_coarse.parameters())

    trainable_parameters += list(model_fine.parameters())
    #trainable_parameters_env += list(model_env_fine.parameters())
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
        model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        model_env_coarse.load_state_dict(checkpoint["model_env_coarse_state_dict"])
        model_env_fine.load_state_dict(checkpoint["model_env_fine_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    i = 0
    with torch.no_grad():
        rgb_coarse, rgb_fine = None, None
        target_ray_values = None

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
        ir_extrinsic_target = ir_poses[img_idx,:,:].to(device)
        ray_origins, ray_directions, cam_origins, cam_directions = get_ray_bundle(
            H, W, focal, pose_target, intrinsic_target
        )
        coords = torch.stack(
            meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
            dim=-1,
        )
        #print(ray_directions.shape)
        
        coords = coords.permute(1,0,2)
        coords = coords.reshape((-1, 2))
        #print(coords)
        
        #assert 1==0
        #rgb_coarse, _, _, rgb_fine, _, _ ,depth_fine_dex
        albedo_edit = torch.load('/code/test/brdf_map/new_albedo_map.pt').cuda()
        roughness_edit = torch.load('/code/test/brdf_map/new_roughness_map.pt').cuda()
        print(normal_target.shape)
        assert 1==0
        normal_edit = F.normalize(normal_target.view([129600, 3]), p=2, dim=-1)

        #print(torch.norm(normal_edit, dim=-1))
        #assert 1==0

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
            mode="test",
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            m_thres_cand=m_thres_cand,
            idx = coords,
            albedo_edit=albedo_edit,
            roughness_edit=roughness_edit,
            normal_edit=normal_edit,
            logdir=logdir,
            light_extrinsic=ir_extrinsic_target
        )
        rgb_coarse, rgb_coarse_off, rgb_fine, rgb_fine_off = nerf_out[0], nerf_out[1], nerf_out[4], nerf_out[5]
        depth_fine_nerf = nerf_out[8]
        normal_fine, albedo_fine, roughness_fine = nerf_out[10], nerf_out[11], nerf_out[12]
        #print(albedo_fine.shape)
        #assert 1==0
        #normals_diff_map = nerf_out[13]
        depth_fine_dex = list(nerf_out[18:])
        target_ray_values = img_target.unsqueeze(-1)

        coarse_loss = 0.#img2mse(rgb_coarse, target_ray_values)
        loss, fine_loss = 0.0, 0.0
        if rgb_fine is not None:
            fine_loss = img2mse(rgb_fine, target_ray_values)
            loss = fine_loss
        else:
            loss = coarse_loss
        
        #loss = coarse_loss + fine_loss
        
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
        with open(os.path.join(logdir, "output_result.yml"), "a") as f:
            f.write("iter: "
            + str(i)
            + " Validation loss: "
            + str(loss.item())
            + " Validation PSNR: "
            + str(psnr)
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
            + "\n"
            )

if __name__ == "__main__":
    main()