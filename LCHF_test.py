import os
import sys
import time
import numpy as np
import cv2
import math
from pysixd import view_sampler, inout, misc
from pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cxxLCHF_pybind

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

mode = 'render_train'
# mode = 'test'

base_path = join(dp['base_path'], 'LCHF')
train_from_radius = 500
if mode == 'render_train':
    start_time = time.time()
    visual = True
    misc.ensure_dir(base_path)

    ssaa_fact = 4
    im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['cam']['im_size']]
    K_rgb = dp['cam']['K'] * ssaa_fact

    LCHF_infos = []
    LCHF_linemod_feats = []
    for obj_id in obj_ids_curr:
        radii = [train_from_radius]
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
                                                           tilt_range=(-math.pi/2, math.pi/2), tilt_step=0.2*math.pi)
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

                rows_any = np.any(depth, axis=1)
                cols_any = np.any(depth, axis=0)
                ymin, ymax = np.where(rows_any)[0][[0, -1]]
                xmin, xmax = np.where(cols_any)[0][[0, -1]]

                mask = (depth > 0).astype(np.uint8) * 255

                padding = 3
                ymin = ymin - padding
                ymax = ymax + padding
                xmin = xmin - padding
                xmax = xmax + padding

                rgb = rgb[ymin:ymax, xmin:xmax, :]
                depth = depth[ymin:ymax, xmin:xmax]
                mask = mask[ymin:ymax, xmin:xmax]

                rows = depth.shape[0]
                cols = depth.shape[1]
                # have read rgb, depth, pose, obj_bb, obj_id, bbox, mask here

                # 5x5 cm patch, stride 5, assume 1pix = 1mm in around 500mm depth
                stride = 10
                for row in range(0, rows - 50, stride):
                    for col in range(0, cols - 50, stride):
                        offset1 = [col, row, 50, 50]
                        rgb1 = rgb[offset1[1]:(offset1[1] + offset1[3]), offset1[0]:(offset1[0] + offset1[2]), :]
                        depth1 = depth[offset1[1]:(offset1[1] + offset1[3]), offset1[0]:(offset1[0] + offset1[2])]

                        visualized = False
                        if visualized:
                            rgb_ = np.copy(rgb)
                            cv2.rectangle(rgb_, (offset1[0], offset1[1]),
                                          (offset1[0] + offset1[2], offset1[1] + offset1[3]), (0, 0, 255), 1)
                            cv2.imshow('rgb', rgb_)
                            cv2.imshow('rgb1', rgb1)
                            cv2.waitKey(0)

                        LCHF_linemod_feat = cxxLCHF_pybind.Linemod_feature(rgb1, depth1)
                        if LCHF_linemod_feat.constructEmbedding():  # extract template OK
                            LCHF_linemod_feat.constructResponse()  # extract response map for simi func
                        else:
                            # print('points not enough')
                            continue  # no enough points for template extraction, pass

                        LCHF_linemod_feats.append(LCHF_linemod_feat)  # record feature

                        LCHF_info = cxxLCHF_pybind.Info()
                        LCHF_info.rpy = (rotationMatrixToEulerAngles(R)).astype(np.float32)  # make sure consistent
                        LCHF_info.t = (np.array(offset1)).astype(np.float32)
                        LCHF_info.id = str(obj_id)
                        LCHF_infos.append(LCHF_info)  # record info

                del rgb, depth, mask

    elapsed_time = time.time() - start_time
    print('construct features time: {}\n'.format(elapsed_time))

    print('sample size: {}\n'.format(len(LCHF_linemod_feats)))
    cxxLCHF_pybind.lchf_model_saveInfos(LCHF_infos, base_path)
    cxxLCHF_pybind.lchf_model_saveFeatures(LCHF_linemod_feats, base_path)

    forest = cxxLCHF_pybind.lchf_model_train(LCHF_linemod_feats, LCHF_infos)
    cxxLCHF_pybind.lchf_model_saveForest(forest, base_path)

    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'test':
    print('reading detector forest & info')

    LCHF_infos = cxxLCHF_pybind.lchf_model_loadInfos(base_path)
    LCHF_linemod_feats = cxxLCHF_pybind.lchf_model_loadFeatures(base_path)

    forest = cxxLCHF_pybind.lchf_model_loadForest(base_path)
    leaf_feats_map = cxxLCHF_pybind.getLeaf_feats_map(forest)

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

            rows = depth.shape[0]
            cols = depth.shape[1]
            stride = 5

            # should be max_bbox * render_depth/max_scene_depth
            width = 50  # bigger is OK, top left corner should align obj
            height = 50
            dep_x = 10  # closer to top left, for patch depth estimation
            dep_y = 10

            start_time = time.time()

            rois = []
            for x in range(0, cols - width - 2*stride, stride):  # avoid out of img
                for y in range(0, rows - height - 2*stride, stride):

                    ker_size = 5
                    dep_sum = 0
                    dep_valid = 0
                    for i in range(ker_size):
                        for j in range(ker_size):
                            depth_value = depth[i + dep_y + y, j + dep_x + x]
                            if depth_value > 0:
                                dep_sum += depth_value
                                dep_valid += 1
                    if dep_valid > 0:
                        dep_sum /= dep_valid
                    else:
                        continue

                    roi = [x, y, width, height, int(dep_sum)]
                    rois.append(roi)

            scene_feats = cxxLCHF_pybind.get_feats_from_scene(rgb, depth, rois)
            leaf_of_trees_of_scene = cxxLCHF_pybind.lchf_model_predict(forest, LCHF_linemod_feats, scene_feats)

            elapsed_time = time.time() - start_time
            start_time = time.time()
            print('forest predict time: {}'.format(elapsed_time))

            steps = 10
            num_x_bins = int(cols/steps)
            num_y_bins = int(rows/steps)
            num_angle_bins = 10

            print('x_bins: {}, y_bins: {}'.format(num_x_bins, num_y_bins))

            votes = np.zeros(shape=(num_x_bins, num_y_bins, num_angle_bins, num_angle_bins, num_angle_bins),
                             dtype=np.float32)

            voted_ids = {}

            for scene_i in range(len(leaf_of_trees_of_scene)):
                trees_of_scene = leaf_of_trees_of_scene[scene_i]
                roi = rois[scene_i]

                for tree_i in range(len(trees_of_scene)):
                    leaf_i = trees_of_scene[tree_i]

                    # if leaf_i has predicted
                    if (tree_i, leaf_i) in voted_ids:
                        votes += voted_ids[(tree_i, leaf_i)]
                    else:
                        # leaf_i votes
                        votes_local = np.zeros(
                            shape=(num_x_bins, num_y_bins, num_angle_bins, num_angle_bins, num_angle_bins),
                            dtype=np.float32)

                        leaf_map = leaf_feats_map[tree_i]
                        predicted_ids = leaf_map[leaf_i]
                        for id_ in predicted_ids:
                            info = LCHF_infos[id_]
                            offset = info.t
                            offset_x = offset[0] * train_from_radius / roi[4]
                            offset_y = offset[1] * train_from_radius / roi[4]

                            x = int((roi[0] - offset_x) / steps)
                            y = int((roi[1] - offset_y) / steps)
                            theta0 = int(info.rpy[0] / 2 / 3.14 * num_angle_bins)
                            theta1 = int(info.rpy[1] / 2 / 3.14 * num_angle_bins)
                            theta2 = int(info.rpy[2] / 2 / 3.14 * num_angle_bins)

                            # votes[x-1:x+1, y-1:y+1, theta0-1:theta0+1, theta1-1:theta1+1, theta2-1:theta2+1] \
                            #     += 1.0/len(predicted_ids)/len(trees_of_scene)
                            votes_local[x, y, theta0, theta1, theta2] \
                                += 1.0 / len(predicted_ids) / len(trees_of_scene)
                            votes += votes_local

                            # cache
                            voted_ids[(tree_i, leaf_i)] = votes_local

            votes_sort_idx = np.dstack(np.unravel_index(np.argsort(votes.ravel()), votes.shape))

            top10 = 10
            if top10>votes_sort_idx.shape[1]:
                top10 = votes_sort_idx.shape[1]

            print('top {}'.format(top10))
            for i in range(1, top10):
                    cv2.circle(rgb, (votes_sort_idx[0, -i, 0]*steps, votes_sort_idx[0, -i, 1]*steps), 4, (0, 255-i*2, 0), -1)

            elapsed_time = time.time() - start_time
            print('voting time: {}'.format(elapsed_time))

            visual = True
            # visual = False
            if visual:
                cv2.namedWindow('rgb')
                cv2.imshow('rgb', rgb)
                cv2.waitKey(1000)

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
