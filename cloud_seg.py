import os
import sys
import time
import numpy as np
# import matplotlib.pyplot as plt
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
from pysixd import view_sampler, inout, misc
from pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join

import cxx_3d_seg_pybind

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

# dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
dataset = 'doumanoglou'
# dataset = 'toyotalight'

# set ./params/dataset_params common_base_path correctly
dp = get_dataset_params(dataset)

obj_ids = [1]  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)



scene_ids = [1]  # for each obj
im_ids = []  # obj's img
gt_ids = []  # multi obj in one img

# Whether to consider only the specified subset of images
use_image_subset = True

# Subset of images to be considered
if use_image_subset:
    im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
else:
    im_ids_sets = None

scene_ids_curr = range(1, dp['scene_count'] + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids)

for scene_id in scene_ids_curr:
    # Load scene info and gt poses
    scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
    scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
    model_path = dp['model_mpath'].format(scene_id)
    model = inout.load_ply(model_path)

    # Considered subset of images for the current scene
    if im_ids_sets is not None:
        im_ids_curr = im_ids_sets[scene_id]
    else:
        im_ids_curr = sorted(scene_info.keys())

    if im_ids:
        im_ids_curr = set(im_ids_curr).intersection(im_ids)

    for im_id in im_ids_curr:
        print('scene: {}, im: {}'.format(scene_id, im_id))

        K = scene_info[im_id]['cam_K']
        render_K = K
        # Load the images
        rgb = inout.load_im(dp['test_rgb_mpath'].format(scene_id, im_id))
        depth = inout.load_depth(dp['test_depth_mpath'].format(scene_id, im_id))
        depth = depth.astype(np.uint16)  # [mm]
        # depth *= dp['cam']['depth_scale']  # to [mm]
        im_size = (depth.shape[1], depth.shape[0])

        match_ids = list()
        match_ids.append('{:02d}_template'.format(scene_id))
        start_time = time.time()

        # result = cxx_3d_seg.convex_cloud_seg(rgb, depth, K.astype(np.float32))
        result = cxx_3d_seg_pybind.convex_cloud_seg(rgb, depth, K.astype(np.float32))
        indices = result.getIndices()
        cloud = result.getCloud()
        normal = result.getNormal()

        # just test one seg result, may break because it's not guaranteed as an object mask
        seg_mask = (indices == 3)
        seg_test_cloud = np.zeros_like(cloud)
        seg_test_cloud[seg_mask] = cloud[seg_mask]

        test_pose = cxx_3d_seg_pybind.pose_estimation(seg_test_cloud, model_path)

        render_R = test_pose[0:3, 0:3]
        render_t = test_pose[0:3, 3:4]


        elapsed_time = time.time() - start_time

        # print("pose refine time: {}s".format(elapsed_time))
        render_rgb, render_depth = render(model, im_size, render_K, render_R, render_t, surf_color=[0, 1, 0])
        visible_mask = render_depth < depth
        mask = render_depth > 0
        mask = mask.astype(np.uint8)
        rgb_mask = np.dstack([mask] * 3)
        render_rgb = render_rgb * rgb_mask
        render_rgb = rgb * (1 - rgb_mask) + render_rgb

        draw_axis(rgb, render_R, render_t, render_K)

        visual = True
        # visual = False
        if visual:
            cv2.namedWindow('rgb')
            cv2.imshow('rgb', rgb)
            cv2.namedWindow('rgb_render')
            cv2.imshow('rgb_render', render_rgb)
            cv2.waitKey(0)

        gt_ids_curr = range(len(scene_gt[im_id]))
        if gt_ids:
            gt_ids_curr = set(gt_ids_curr).intersection(gt_ids)
        # for multi objs in one img
        for gt_id in gt_ids_curr:
            gt = scene_gt[im_id][gt_id]
            obj_id = gt['obj_id']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            # have read rgb, depth, pose, obj_bb, obj_id here