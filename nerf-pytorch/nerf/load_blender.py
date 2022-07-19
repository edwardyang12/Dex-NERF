import json
import os

import cv2
import imageio
import numpy as np
import torch
import pickle

def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1, debug=False):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            #testimg = np.array(imageio.imread(fname))
            #print(testimg.shape, np.max(testimg), np.min(testimg))
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        #print(imgs.shape)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    
    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]
    #print(i_split)
    #assert 1==0
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

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
        poses = torch.from_numpy(poses)
        return imgs, poses, render_poses, [H, W, focal], i_split

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 4
        W = W // 4
        focal = focal / 4.0
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
        #TODO for grayscale images manually expand one dimension
        # imgs = imgs.unsqueeze(-1).repeat(1,1,1,3)

    poses = torch.from_numpy(poses)

    return imgs, poses, render_poses, [H, W, focal], i_split

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def load_messytable_data(basedir, half_res=False, testskip=1, debug=False):
    splits = ["train", "val", "test"]
    metas = {}
    #for s in splits:
    #    with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
    #        metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_intrinsics = []
    counts = [0]
    for s in splits:
        path = os.path.join(basedir, s)
        imgs = []
        poses = []
        intrinsics = []
        #print(os.listdir(path))
        for prefix in os.listdir(path):
            #print(os.path.join(path, prefix, 'meta.pkl'))
            meta = load_pickle(os.path.join(path, prefix, 'meta.pkl'))
            
            if s == "train" or testskip == 0:
                skip = 1
            else:
                skip = testskip

            #for frame in meta["frames"][::skip]:
            fname = os.path.join(path, prefix, "0128_rgbL_kuafu.png")
            #testimg = np.array(imageio.imread(fname))
            #print(testimg.shape, np.max(testimg), np.min(testimg))
            imgs.append(imageio.imread(fname))
            poses.append(np.array(meta["extrinsic_l"]))
            if half_res:
                intrinsics_c = np.array(meta['intrinsic_l'])
                intrinsics_c[:2,:] = intrinsics_c[:2,:]/4
                intrinsics_c[0,2] = 240.
                intrinsics_c[1,2] = 135.
                intrinsics.append(intrinsics_c)
            else:
                intrinsics.append(np.array(meta['intrinsic_l']))
            
            
            #print(imgs.shape)
        poses = np.array(poses).astype(np.float32)
        intrinsics = np.array(intrinsics).astype(np.float32)
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        #print(imgs.shape)
        counts.append(counts[-1] + imgs.shape[0])
        #assert 1==0
        
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_intrinsics.append(intrinsics)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    intrinsics = np.concatenate(all_intrinsics, 0)

    H, W = imgs[0].shape[:2]
    #camera_angle_x = float(meta["camera_angle_x"])
    focal = meta['intrinsic_l'][0,0]

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
        poses = torch.from_numpy(poses)
        return imgs, poses, render_poses, [H, W, focal], i_split

    if half_res:
        # TODO: resize images using INTER_AREA (cv2)
        H = H // 4
        W = W // 4
        focal = focal / 4.0
        
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
        #TODO for grayscale images manually expand one dimension
        # imgs = imgs.unsqueeze(-1).repeat(1,1,1,3)

    poses = torch.from_numpy(poses)
    intrinsics = torch.from_numpy(intrinsics)
    #print(i_split)
    #print(poses)
    #print(poses.shape)
    #assert 1==0

    return imgs, poses, render_poses, [H, W, focal], i_split, intrinsics
