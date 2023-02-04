"""
Utils to compute metrics and track them across training.
"""

import torch
import torch.nn.functional as F
import numpy as np
import copy


class ScalarMetric(object):
    def __init__(self):
        self.value = 0.0
        self.num_observations = 0.0
        self.aggregated_value = 0.0
        self.reset()

    def reset(self):
        self.value = []
        self.num_observations = 0.0
        self.aggregated_value = 0.0

    def __repr__(self):
        return str(self.peek())

    def update(self, x):
        self.aggregated_value += x
        self.num_observations += 1

    def peek(self, x):
        return self.aggregated_value / (
            self.num_observations if self.num_observations > 0 else 1
        )


def compute_err_metric(depth_gt, depth_pred, mask):
    """
    Compute the error metrics for predicted disparity map
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param mask: Selected pixel
    :return: Error metrics
    """

    depth_abs_err = F.l1_loss(depth_pred[mask] * 1000, depth_gt[mask] * 1000, reduction='mean').item()
    depth_diff = torch.abs(depth_gt[mask] - depth_pred[mask])  # [bs, 1, H, W]
    depth_err2 = depth_diff[depth_diff > 2e-3].numel() / depth_diff.numel()
    depth_err4 = depth_diff[depth_diff > 4e-3].numel() / depth_diff.numel()
    depth_err8 = depth_diff[depth_diff > 8e-3].numel() / depth_diff.numel()

    err = {}

    err['depth_abs_err'] = depth_abs_err
    err['depth_err2'] = depth_err2
    err['depth_err4'] = depth_err4
    err['depth_err8'] = depth_err8
    return err

def compute_obj_err(depth_gt, depth_pred, label, mask, obj_total_num=17):
    """
    Compute error for each object instance in the scene
    :param disp_gt: GT disparity map, [bs, 1, H, W]
    :param depth_gt: GT depth map, [bs, 1, H, W]
    :param disp_pred: Predicted disparity map, [bs, 1, H, W]
    :param focal_length: Focal length, [bs, 1]
    :param baseline: Baseline of the camera, [bs, 1]
    :param label: Label of the image [bs, 1, H, W]
    :param obj_total_num: Total number of objects in the dataset
    :return: obj_disp_err, obj_depth_err - List of error of each object
             obj_count - List of each object appear count
    """

    obj_list = label.unique()  # TODO this will cause bug if bs > 1, currently only for testing
    obj_num = obj_list.shape[0]

    # Array to store error and count for each object

    total_obj_depth_err = np.zeros(obj_total_num)
    total_obj_depth_4_err = np.zeros(obj_total_num)
    total_obj_count = np.zeros(obj_total_num)

    for i in range(obj_num):
        obj_id = int(obj_list[i].item())
        if obj_id >= 17:
            continue
        obj_mask = label == obj_id

        obj_depth_err = F.l1_loss(depth_gt[obj_mask * mask] * 1000, depth_pred[obj_mask * mask] * 1000,
                                  reduction='mean').item()
        obj_depth_diff = torch.abs(depth_gt[obj_mask * mask] - depth_pred[obj_mask * mask])
        obj_depth_err4 = obj_depth_diff[obj_depth_diff > 4e-3].numel() / obj_depth_diff.numel()


        total_obj_depth_err[obj_id] += obj_depth_err
        total_obj_depth_4_err[obj_id] += obj_depth_err4
        total_obj_count[obj_id] += 1
    return total_obj_depth_err, total_obj_depth_4_err, total_obj_count

def gen_error_colormap_depth():
    cols = np.array(
        [[0, 0.00001, 0, 0, 0],
         [0.00001, 2000./(2**10) , 49, 54, 149],
         [2000./(2**10) , 2000./(2**9) , 69, 117, 180],
         [2000./(2**9) , 2000./(2**8) , 116, 173, 209],
         [2000./(2**8), 2000./(2**7), 171, 217, 233],
         [2000./(2**7), 2000./(2**6), 224, 243, 248],
         [2000./(2**6), 2000./(2**5), 254, 224, 144],
         [2000./(2**5), 2000./(2**4), 253, 174, 97],
         [2000./(2**4), 2000./(2**3), 244, 109, 67],
         [2000./(2**3), 2000./(2**2), 215, 48, 39],
         [2000./(2**2), np.inf, 165, 0, 38]], dtype=np.float32)
    cols[:, 2: 5] /= 255.
    return cols

def depth_error_img(D_est_tensor, D_gt_tensor, mask, abs_thres=1., dilate_radius=1):
    D_gt_np = D_gt_tensor.detach().cpu().numpy()
    D_est_np = D_est_tensor.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    B, H, W = D_gt_np.shape
    # valid mask
    # mask = (D_gt_np > 0) & (D_gt_np < 1250)
    # error in percentage. When error <= 1, the pixel is valid since <= 3px & 5%
    error = np.abs(D_gt_np - D_est_np)
    error[np.logical_not(mask)] = 0
    error[mask] = error[mask] / abs_thres
    # get colormap
    cols = gen_error_colormap_depth()
    # create error image
    error_image = np.zeros([B, H, W, 3], dtype=np.float32)
    for i in range(cols.shape[0]):
        error_image[np.logical_and(error >= cols[i][0], error < cols[i][1])] = cols[i, 2:]
    # error_image = cv2.imdilate(D_err, strel('disk', dilate_radius));
    error_image[np.logical_not(mask)] = 0.
    # show color tag in the top-left cornor of the image
    for i in range(cols.shape[0]):
        distance = 20
        error_image[:, :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
    return error_image[0] # [H, W, 3]