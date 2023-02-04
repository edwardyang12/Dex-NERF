import json
import os

import cv2
import imageio
import numpy as np
import torch
import pickle
import PIL
from PIL import Image
from .load_blender import pose_spherical

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_messytable_data(basedir, half_res=False, testskip=1, debug=False, cfg=None, is_real_rgb = False, sceneid = 1):
    basedir = os.path.join(basedir, str(sceneid))
    imgname = cfg.dataset.imgname
    imgname_off = cfg.dataset.imgname_off
    label_n = cfg.dataset.labelname
    splits = ["train", "val", "test"]
    metas = {}

    all_imgs = []
    all_poses = []
    all_intrinsics = []
    all_depths = []
    all_labels = []
    all_imgs_off = []
    all_normals = []
    counts = [0]
    #is_real_rgb = False
    if is_real_rgb:
        depth_n = "depth.png"
        extri_n = "extrinsic"
        intri_n = "intrinsic"
    else:
        depth_n = "depthL.png"
        extri_n = "extrinsic_l"
        intri_n = "intrinsic_l"

    for s in splits:
        if debug:
            test = True if s != "train" else False
            s = splits[0]
            
        path = os.path.join(basedir, s)
        
        imgs = []
        poses = []
        intrinsics = []
        depths = []
        labels = []
        imgs_off = []
        normals = []
        idx = 0
        for prefix in os.listdir(path):
            
            meta = load_pickle(os.path.join(path, prefix, 'meta.pkl'))
            
            if s == "train" or testskip == 0:
                skip = 1
            else:
                skip = testskip

            fname = os.path.join(path, prefix, imgname)
            fname_off = os.path.join(path, prefix, imgname_off)
            gt_depth_fname = os.path.join(path, prefix, depth_n)
            label_fname = os.path.join(path, prefix, label_n)
            normal_fname = os.path.join(path, prefix, "normalL.png")

            cur_img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
            cur_img_off = cv2.imread(fname_off, cv2.IMREAD_UNCHANGED)
            normal_img = cv2.imread(normal_fname, cv2.IMREAD_UNCHANGED)
            normal_img = (normal_img.astype(float)) / 1000 - 1
            
            H,W = cur_img.shape[:2]
            imgs.append(cur_img)
            depths.append(np.array(Image.open(gt_depth_fname))/1000)
            poses.append(np.array(meta[extri_n]))
            labels.append(np.array(Image.open(label_fname)))
            imgs_off.append(cur_img_off)
            normals.append(normal_img)
            if half_res:
                intrinsics_c = np.array(meta[intri_n])
                intrinsics_c[:2,:] = intrinsics_c[:2,:]/4
                intrinsics.append(intrinsics_c)
            else:
                intrinsics.append(np.array(meta[intri_n]))
            
            if debug:
                if idx == 0:
                    break
            idx += 1
            
        poses = np.array(poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        depths = np.array(depths).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        imgs_off = (np.array(imgs_off) / 255.0).astype(np.float32)
        normals = (np.array(normals)).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_depths.append(depths)
        all_labels.append(labels)
        all_imgs_off.append(imgs_off)
        all_normals.append(normals)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    depths = np.concatenate(all_depths,0)
    labels = np.concatenate(all_labels,0)
    imgs_off = np.concatenate(all_imgs_off,0)
    normals = np.concatenate(all_normals,0)

    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    focal = meta[intri_n][0,0]

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = 270
        W = 480
        focal = focal / 4.0
    else:
        H = 1080
        W = 1920
        
    imgs = [
        torch.from_numpy(
            cv2.resize(imgs[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs.shape[0])
    ]
    imgs = torch.stack(imgs, 0)

    imgs_off = [
        torch.from_numpy(
            cv2.resize(imgs_off[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs_off.shape[0])
    ]
    imgs_off = torch.stack(imgs_off, 0)

    depths = [
        torch.from_numpy(
            cv2.resize(depths[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(depths.shape[0])
    ]
    depths = torch.stack(depths, 0)

    normals = [
        torch.from_numpy(
            cv2.resize(normals[i], dsize=(W, H), interpolation=PIL.Image.NEAREST)
        )
        for i in range(normals.shape[0])
    ]
    normals = torch.stack(normals, 0)

    labels = [
        torch.from_numpy(
            cv2.resize(labels[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(labels.shape[0])
    ]
    labels = torch.stack(labels, 0)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    return imgs, poses, render_poses, [H, W, focal], i_split, intrinsics, depths, labels, imgs_off, normals


def load_messytable_data_RF(basedir, half_res=False, testskip=1, debug=False, cfg=None, is_real_rgb = False, sceneid = 1):
    basedir = os.path.join(basedir, str(sceneid))
    imgname = cfg.dataset.imgname
    imgname_off = cfg.dataset.imgname_off
    label_n = cfg.dataset.labelname
    splits = ["train", "val", "test"]
    metas = {}

    all_imgs = []
    all_poses = []
    all_intrinsics = []
    all_depths = []
    all_labels = []
    all_imgs_off = []
    counts = [0]
    all_ids = []
    #is_real_rgb = False
    if is_real_rgb:
        depth_n = "depth.png"
        extri_n = "extrinsic"
        intri_n = "intrinsic"
    else:
        depth_n = "depthL.png"
        extri_n = "extrinsic_l"
        intri_n = "intrinsic_l"

    for s in splits:
        if debug:
            test = True if s != "train" else False
            s = splits[0]
            
        path = os.path.join(basedir, s)
        
        imgs = []
        poses = []
        intrinsics = []
        depths = []
        labels = []
        imgs_off = []
        ids = []
        idx = 0
        for prefix in os.listdir(path):
            
            scene_id = int(prefix.split('-')[-1])
            meta = load_pickle(os.path.join(path, prefix, 'meta.pkl'))
            
            if s == "train" or testskip == 0:
                skip = 1
            else:
                skip = testskip

            fname = os.path.join(path, prefix, imgname)
            fname_off = os.path.join(path, prefix, imgname_off)
            gt_depth_fname = os.path.join(path, prefix, depth_n)
            label_fname = os.path.join(path, prefix, label_n)
            cur_img = imageio.imread(fname).astype(int)
            cur_img_off = imageio.imread(fname_off).astype(int)
            H,W = cur_img.shape[:2]
            imgs.append(cur_img)
            depths.append(np.array(Image.open(gt_depth_fname))/1000)
            poses.append(np.array(meta[extri_n]))
            labels.append(np.array(Image.open(label_fname)))
            imgs_off.append(cur_img_off)
            ids.append(scene_id)
            if half_res:
                intrinsics_c = np.array(meta[intri_n])
                intrinsics_c[:2,:] = intrinsics_c[:2,:]/4
                intrinsics.append(intrinsics_c)
            else:
                intrinsics.append(np.array(meta[intri_n]))
            
            if debug:
                if idx == 0:
                    break
            idx += 1
            
        poses = np.array(poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        depths = np.array(depths).astype(np.float32)
        labels = np.array(labels).astype(np.float32)
        imgs_off = (np.array(imgs_off) / 255.0).astype(np.float32)
        ids = np.array(ids).astype(int)
        counts.append(counts[-1] + imgs.shape[0])
        
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_depths.append(depths)
        all_labels.append(labels)
        all_imgs_off.append(imgs_off)
        all_ids.append(ids)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    depths = np.concatenate(all_depths,0)
    labels = np.concatenate(all_labels,0)
    imgs_off = np.concatenate(all_imgs_off,0)
    ids = np.concatenate(all_ids,0)

    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    focal = meta[intri_n][0,0]

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )



    if half_res:
        H = 270
        W = 480
        focal = focal / 4.0
    else:
        H = 1080
        W = 1920
        
    imgs = [
        torch.from_numpy(
            cv2.resize(imgs[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs.shape[0])
    ]
    imgs = torch.stack(imgs, 0)

    imgs_off = [
        torch.from_numpy(
            cv2.resize(imgs_off[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs_off.shape[0])
    ]
    imgs_off = torch.stack(imgs_off, 0)

    depths = [
        torch.from_numpy(
            cv2.resize(depths[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(depths.shape[0])
    ]
    depths = torch.stack(depths, 0)

    labels = [
        torch.from_numpy(
            cv2.resize(labels[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(labels.shape[0])
    ]
    labels = torch.stack(labels, 0)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)

    return imgs, poses, render_poses, [H, W, focal], i_split, intrinsics, depths, labels, imgs_off, ids

