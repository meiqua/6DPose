import os
import sys
import time
import numpy as np
# import matplotlib.pyplot as plt
import cv2
import math
from pysixd import view_sampler, inout, misc
from pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join

import cxxLCHF_pybind


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def draw_axis(img, R, t, K):
    # unit is mm
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

# set ./params/dataset_params common_base_path correctly
dp = get_dataset_params(dataset)

obj_ids = [6]  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

scene_ids = [6]  # for each obj
im_ids = []  # obj's img
gt_ids = []  # multi obj in one img
scene_ids_curr = range(1, dp['scene_count'] + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids)

# mode = 'render_train'
mode = 'test'

template_saved_to = join(dp['base_path'], 'LCHF', '%s.yaml')
tempInfo_saved_to = join(dp['base_path'], 'LCHF', '{:02d}_info.yaml')
if mode == 'train':
    start_time = time.time()
    # im_ids = list(range(1, 1000, 10))  # obj's img
    im_ids = []
    visual = True
    misc.ensure_dir(os.path.dirname(template_saved_to))

    for obj_id in obj_ids_curr:
        scene_info = inout.load_info(dp['obj_info_mpath'].format(obj_id))
        scene_gt = inout.load_gt(dp['obj_gt_mpath'].format(obj_id))

        im_ids_curr = sorted(scene_info.keys())

        if im_ids:
            im_ids_curr = set(im_ids_curr).intersection(im_ids)

        templateInfo = dict()
        for im_id in im_ids_curr:
            print('obj: {}, im: {}'.format(obj_id, im_id))

            # Load the images
            rgb = inout.load_im(dp['train_rgb_mpath'].format(obj_id, im_id))
            depth = inout.load_depth(dp['train_depth_mpath'].format(obj_id, im_id))

            depth *= dp['cam']['depth_scale']  # to [mm]
            depth = depth.astype(np.uint16)  # [mm]

            # during training, there's only one obj
            gt = scene_gt[im_id][0]

            K = scene_info[im_id]['cam_K']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            # have read rgb, depth, pose, obj_bb, obj_id here

            aTemplateInfo = dict()
            aTemplateInfo['cam_K'] = K
            aTemplateInfo['cam_R_w2c'] = R
            aTemplateInfo['cam_t_w2c'] = t

            mask = (depth > 0).astype(np.uint8) * 255

            # visual = False
            if visual:
                cv2.namedWindow('rgb')
                cv2.imshow('rgb', rgb)
                cv2.namedWindow('depth')
                cv2.imshow('depth', depth)
                cv2.namedWindow('mask')
                cv2.imshow('mask', mask)
                cv2.waitKey(1)

    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'render_train':
    start_time = time.time()
    visual = True
    misc.ensure_dir(os.path.dirname(template_saved_to))

    ssaa_fact = 4
    im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['cam']['im_size']]
    K_rgb = dp['cam']['K'] * ssaa_fact

    for obj_id in obj_ids_curr:
        templateInfo = dict()

        radii = [1000]
        azimuth_range = (0, 2 * math.pi)
        elev_range = (0, 0.5 * math.pi)
        min_n_views = 100
        clip_near = 10  # [mm]
        clip_far = 10000  # [mm]
        ambient_weight = 0.8  # Weight of ambient light [0, 1]
        shading = 'phong'  # 'flat', 'phong'

        # Load model
        model_path = dp['model_mpath'].format(obj_id)
        model = inout.load_ply(model_path)

        # Load model texture
        if dp['model_texture_mpath']:
            model_texture_path = dp['model_texture_mpath'].format(obj_id)
            model_texture = inout.load_im(model_texture_path)
        else:
            model_texture = None

        for radius in radii:
            # Sample views
            views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                           azimuth_range, elev_range,
                                                           tilt_range=(0, 2*math.pi), tilt_step=0.1*math.pi)
            print('Sampled views: ' + str(len(views)))

            # Render the object model from all the views
            for view_id, view in enumerate(views):
                if view_id % 10 == 0:
                    print('obj,radius,view: ' + str(obj_id) +
                          ',' + str(radius) + ',' + str(view_id))

                # Render depth image
                depth = render(model, dp['cam']['im_size'], dp['cam']['K'],
                                        view['R'], view['t'],
                                        clip_near, clip_far, mode='depth')

                # Convert depth so it is in the same units as the real test images
                depth /= dp['cam']['depth_scale']
                depth = depth.astype(np.uint16)

                # Render RGB image
                rgb = render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                      clip_near, clip_far, texture=model_texture,
                                      ambient_weight=ambient_weight, shading=shading,
                                      mode='rgb')
                rgb = cv2.resize(rgb, dp['cam']['im_size'], interpolation=cv2.INTER_AREA)

                K = dp['cam']['K']
                R = view['R']
                t = view['t']
                # have read rgb, depth, pose, obj_bb, obj_id here

                rows = np.any(depth, axis=1)
                cols = np.any(depth, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                # cv2.rectangle(rgb, (xmin, ymin), (xmax, ymax),(0,255,0),3)
                # cv2.imshow('mask', rgb)
                # cv2.waitKey(0)

                aTemplateInfo = dict()
                aTemplateInfo['cam_K'] = K
                aTemplateInfo['cam_R_w2c'] = R
                aTemplateInfo['cam_t_w2c'] = t
                aTemplateInfo['width'] = int(xmax-xmin)
                aTemplateInfo['height'] = int(ymax-ymin)

                mask = (depth > 0).astype(np.uint8) * 255

                # visual = False
                if visual:
                    cv2.namedWindow('rgb')
                    cv2.imshow('rgb', rgb)
                    # cv2.namedWindow('depth')
                    # cv2.imshow('depth', depth)
                    # cv2.namedWindow('mask')
                    # cv2.imshow('mask', mask)
                    cv2.waitKey(1)

                del rgb, depth, mask

    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'test':
    print('reading detector template & info')
    templateInfo = dict()

    # Whether to consider only the specified subset of images
    use_image_subset = True

    # Subset of images to be considered
    if use_image_subset:
        im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
    else:
        im_ids_sets = None

    for scene_id in scene_ids_curr:
        # Load scene info and gt poses
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
        model = inout.load_ply(dp['model_mpath'].format(scene_id))
        aTemplateInfo = inout.load_info(tempInfo_saved_to.format(scene_id))

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
            # Load the images
            rgb = inout.load_im(dp['test_rgb_mpath'].format(scene_id, im_id))
            depth = inout.load_depth(dp['test_depth_mpath'].format(scene_id, im_id))
            depth *= dp['cam']['depth_scale']  # to [mm]
            depth = depth.astype(np.uint16)  # [mm]
            im_size = (depth.shape[1], depth.shape[0])

            match_ids = list()
            match_ids.append('{:02d}_template'.format(scene_id))
            start_time = time.time()

            elapsed_time = time.time() - start_time

            visual = True
            # visual = False
            if visual:
                cv2.namedWindow('rgb')
                cv2.imshow('rgb', rgb)
                cv2.waitKey(2000)

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

print('end line for debug')
