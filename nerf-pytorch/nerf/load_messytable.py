import json
import os

import cv2
import imageio
import numpy as np
import torch
import pickle
from PIL import Image
from .load_blender import pose_spherical

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_messytable_data(basedir, half_res=False, testskip=1, debug=False, imgname="0128_irL_kuafu_half.png"):
    splits = ["train", "val", "test"]
    metas = {}
    #for s in splits:
    #    with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
    #        metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_intrinsics = []
    all_depths = []
    counts = [0]
    is_real_rgb = True
    if is_real_rgb:
        depth_n = "depth.png"
        extri_n = "extrinsic"
        intri_n = "intrinsic"
    else:
        depth_n = "depthL.png"
        extri_n = "extrinsic_l"
        intri_n = "intrinsic_l"
    for s in splits:
        path = os.path.join(basedir, s)
        imgs = []
        poses = []
        intrinsics = []
        depths = []
        #print(os.listdir(path))
        for prefix in os.listdir(path):
            #print(os.path.join(path, prefix, 'meta.pkl'))
            meta = load_pickle(os.path.join(path, prefix, 'meta.pkl'))

            if s == "train" or testskip == 0:
                skip = 1
            else:
                skip = testskip

            #for frame in meta["frames"][::skip]:
            fname = os.path.join(path, prefix, imgname)
            gt_depth_fname = os.path.join(path, prefix, depth_n)
            #testimg = np.array(imageio.imread(fname))
            #print(testimg.shape, np.max(testimg), np.min(testimg))
            cur_img = imageio.imread(fname)
            if len(cur_img.shape) != 3:
                cur_img = np.array(cur_img)[...,None]
                cur_img = np.concatenate((cur_img, cur_img, cur_img), axis=-1)
                #print(cur_img.shape)
            H,W = cur_img.shape[:2]
            #print(cur_img.shape)
            #assert 1==0
            imgs.append(cur_img)
            depths.append(np.array(Image.open(gt_depth_fname))/1000)
            poses.append(np.array(meta[extri_n]))
            if half_res:
                intrinsics_c = np.array(meta[intri_n])
                intrinsics_c[:2,:] = intrinsics_c[:2,:]/4
                intrinsics_c[0,2] = 240.
                intrinsics_c[1,2] = 135.
                intrinsics.append(intrinsics_c)
            else:
                intrinsics.append(np.array(meta[intri_n]))


            #print(imgs.shape)
        poses = np.array(poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        depths = np.array(depths).astype(np.float32)
        #print(imgs.shape)
        counts.append(counts[-1] + imgs.shape[0])
        #assert 1==0

        all_imgs.append(imgs)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)
        all_depths.append(depths)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)
    depths = np.concatenate(all_depths,0)

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

    # In debug mode, return extremely tiny images
    if debug:
        H = H // 32
        W = W // 32
        focal = focal / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

        depths = [
            torch.from_numpy(
                cv2.resize(depths[i], dsize=(25, 25), interpolation=cv2.INTER_NEAREST)
            )
            for i in range(depths.shape[0])
        ]
        depths = torch.stack(depths, 0)

        poses = torch.from_numpy(poses)
        intrinsics = torch.from_numpy(intrinsics)
        return imgs, poses, render_poses, [H, W, focal], i_split, intrinsics, depths

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = 270
        W = 480
        focal = focal / 4.0
        #print(H,W)
    else:
        H = 1080
        W = 1920

    # H = H // 4
    # W = W // 4
    # focal = focal / 4.0
    imgs = [
        torch.from_numpy(
            cv2.resize(imgs[i], dsize=(W, H), interpolation=cv2.INTER_AREA)
        )
        for i in range(imgs.shape[0])
    ]
    imgs = torch.stack(imgs, 0)

    depths = [
        torch.from_numpy(
            cv2.resize(depths[i], dsize=(W, H), interpolation=cv2.INTER_NEAREST)
        )
        for i in range(depths.shape[0])
    ]
    depths = torch.stack(depths, 0)
        #TODO for grayscale images manually expand one dimension
        # imgs = imgs.unsqueeze(-1).repeat(1,1,1,3)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)
    #print(i_split)
    #print(poses)
    #print(poses.shape)
    #assert 1==0

    return imgs, poses, render_poses, [H, W, focal], i_split, intrinsics, depths
