import os
import sys
import time
import numpy as np
import cv2
import math
from pysixd import view_sampler, inout, misc
from  pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join
import copy
import linemodLevelup_pybind

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

# dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
dataset = 'doumanoglou'
# dataset = 'toyotalight'

# mode = 'render_train'
mode = 'test'

dp = get_dataset_params(dataset)
detector = linemodLevelup_pybind.Detector(16, [4, 8], 16)  # min features; pyramid strides; num clusters

obj_ids = []  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

scene_ids = []  # for each obj
im_ids = []  # obj's img
gt_ids = []  # multi obj in one img
scene_ids_curr = range(1, dp['scene_count'] + 1)
if scene_ids:
    scene_ids_curr = set(scene_ids_curr).intersection(scene_ids)

# mm
dep_range = 200  # max depth range of objects
dep_anchors = []  # depth to apply templates

dep_min = dp['test_obj_depth_range'][0]  # min depth of scene
dep_max = dp['test_obj_depth_range'][1]  # max depth of scene
dep_anchor_step = 1.2  # depth scale
# dep_min = 600  # min depth of scene
# dep_max = 1400  # max depth of scene
# dep_anchor_step = 1.1  # depth scale

current_dep = dep_min
while current_dep < dep_max:
    dep_anchors.append(int(current_dep))
    current_dep = current_dep*dep_anchor_step

print('\ndep anchors:\n {}, \ndep range: {}\n'.format(dep_anchors, dep_range))

top_level_path = os.path.dirname(os.path.abspath(__file__))
template_saved_to = join(dp['base_path'], 'linemod_render_up', '%s.yaml')
tempInfo_saved_to = join(dp['base_path'], 'linemod_render_up', '{:02d}_info_{}.yaml')
result_base_path = join(top_level_path, 'public', 'sixd_results', 'patch-linemod_'+dataset)

misc.ensure_dir(os.path.dirname(template_saved_to))
misc.ensure_dir(os.path.dirname(tempInfo_saved_to))
misc.ensure_dir(result_base_path)

if mode == 'render_train':
    start_time = time.time()
    visual = True

    ssaa_fact = 4
    im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['cam']['im_size']]
    K_rgb = dp['cam']['K'] * ssaa_fact

    for obj_id in obj_ids_curr:
        azimuth_range = dp['test_obj_azimuth_range']
        elev_range = dp['test_obj_elev_range']
        min_n_views = 200
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

        fast_train = True  # just scale templates

        if fast_train:
            # Sample views

            # with camera tilt
            views, views_level = view_sampler.sample_views(min_n_views, dep_anchors[0],
                                                           azimuth_range, elev_range,
                                                           tilt_range=(-math.pi, math.pi),
                                                           tilt_step=math.pi / 8)

            print('Sampled views: ' + str(len(views)))

            templateInfo_radius = dict()
            for dep in dep_anchors:
                templateInfo_radius[dep] = dict()

            # Render the object model from all the views
            for view_id, view in enumerate(views):

                if view_id % 10 == 0:
                    print('obj,radius,view: ' + str(obj_id) +
                          ',' + str(dep_anchors[0]) + ',' + str(view_id) + ', view_id: ', view_id)

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

                # have read rgb, depth, pose, obj_bb, obj_id here

                rows = np.any(depth, axis=1)
                cols = np.any(depth, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]

                mask = (depth > 0).astype(np.uint8) * 255

                visual = False
                if visual:
                    cv2.namedWindow('rgb')
                    cv2.imshow('rgb', rgb)
                    cv2.waitKey(1000)

                success = detector.addTemplate([rgb, depth], '{:02d}_template_{}'.format(obj_id, dep_anchors[0]),
                                               mask, dep_anchors)
                del rgb, depth, mask

                print('success: {}'.format(success))
                for i in range(len(dep_anchors)):
                    if success[i] != -1:
                        aTemplateInfo = dict()
                        aTemplateInfo['cam_K'] = copy.deepcopy(dp['cam']['K'])
                        aTemplateInfo['cam_R_w2c'] = copy.deepcopy(view['R'])
                        aTemplateInfo['cam_t_w2c'] = copy.deepcopy(view['t'])
                        aTemplateInfo['cam_t_w2c'][2] = dep_anchors[i]

                        templateInfo = templateInfo_radius[dep_anchors[i]]
                        templateInfo[success[i]] = aTemplateInfo

            for radius in dep_anchors:
                inout.save_info(tempInfo_saved_to.format(obj_id, radius), templateInfo_radius[radius])

            detector.writeClasses(template_saved_to)
            #  clear to save RAM
            detector.clear_classes()
        else:
            for radius in dep_anchors:
                # Sample views

                # with camera tilt
                views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                               azimuth_range, elev_range,
                                                               tilt_range=(-math.pi * (80 / 180), math.pi * (80 / 180)),
                                                               tilt_step=math.pi / 8)
                print('Sampled views: ' + str(len(views)))

                templateInfo = dict()

                # Render the object model from all the views
                for view_id, view in enumerate(views):

                    if view_id % 10 == 0:
                        print('obj,radius,view: ' + str(obj_id) +
                              ',' + str(radius) + ',' + str(view_id) + ', view_id: ', view_id)

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
                    aTemplateInfo['width'] = int(xmax - xmin)
                    aTemplateInfo['height'] = int(ymax - ymin)

                    mask = (depth > 0).astype(np.uint8) * 255

                    visual = False
                    if visual:
                        cv2.namedWindow('rgb')
                        cv2.imshow('rgb', rgb)
                        cv2.waitKey(1000)

                    success = detector.addTemplate([rgb, depth], '{:02d}_template_{}'.format(obj_id, radius), mask, [])
                    print('success {}'.format(success[0]))
                    del rgb, depth, mask

                    if success[0] != -1:
                        templateInfo[success[0]] = aTemplateInfo

                inout.save_info(tempInfo_saved_to.format(obj_id, radius), templateInfo)
                detector.writeClasses(template_saved_to)
                #  clear to save RAM
                detector.clear_classes()

    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'test':
    print('reading detector template & info')

    use_image_subset = True
    if use_image_subset:
        im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
    else:
        im_ids_sets = None

    for scene_id in scene_ids_curr:
        # Load scene info and gt poses
        misc.ensure_dir(join(result_base_path, '{:02d}'.format(scene_id)))
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))
        model = inout.load_ply(dp['model_mpath'].format(scene_id))

        template_read_classes = []
        detector.clear_classes()
        for radius in dep_anchors:
            template_read_classes.append('{:02d}_template_{}'.format(scene_id, radius))
        detector.readClasses(template_read_classes, template_saved_to)

        templateInfo = dict()
        for radius in dep_anchors:
            key = tempInfo_saved_to.format(scene_id, radius)
            aTemplateInfo = inout.load_info(key)
            key = os.path.basename(key)
            key = os.path.splitext(key)[0]
            key = key.replace('info', 'template')
            templateInfo[key] = aTemplateInfo

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
            depth = depth.astype(np.uint16)  # [mm]
            # depth *= dp['cam']['depth_scale']  # to [mm]
            im_size = (depth.shape[1], depth.shape[0])

            match_ids = list()

            for radius in dep_anchors:
                match_ids.append('{:02d}_template_{}'.format(scene_id, radius))

            start_time = time.time()
            matches = detector.match([rgb, depth], 66.6, 0.66, match_ids, dep_anchors, dep_range, masks=[])
            matching_time = time.time() - start_time

            print('matching time: {}s'.format(matching_time))

            if len(matches) > 0:
                aTemplateInfo = templateInfo[matches[0].class_id]
                render_K = aTemplateInfo[0]['cam_K']

            dets = np.zeros(shape=(len(matches), 5))
            for i in range(len(matches)):
                match = matches[i]
                templ = detector.getTemplates(match.class_id, match.template_id)
                dets[i, 0] = match.x
                dets[i, 1] = match.y
                dets[i, 2] = match.x + templ[0].width
                dets[i, 3] = match.y + templ[0].height
                dets[i, 4] = match.similarity
            idx = nms(dets, 0.4)
            #
            # idx = range(len(matches))  # shouldn't nms here? because of different pose candidates in one position
                                       # nms after locally pose refine
            print('candidates size: {}\n'.format(len(idx)))

            render_rgb = rgb
            color_list = list()
            color_list.append([1, 0, 0])  # blue
            color_list.append([0, 1, 0])  # green
            color_list.append([0, 0, 1])  # red

            color_list.append([0, 1, 1])  # who knows
            color_list.append([1, 0, 1])
            color_list.append([1, 1, 0])

            top5 = 5
            if top5 > len(color_list):
                top5 = len(color_list)
            if top5 > len(idx):
                top5 = len(idx)

            result = {}
            result_ests = []
            result_name = join(result_base_path, '{:02d}'.format(scene_id),'{:04d}_{:02d}.yml'.format(im_id, scene_id))
            for i in reversed(range(top5)):  # avoid overlap high score
                match = matches[idx[i]]
                startPos = (int(match.x), int(match.y))
                aTemplateInfo = templateInfo[match.class_id]
                K_match = aTemplateInfo[match.template_id]['cam_K']
                R_match = aTemplateInfo[match.template_id]['cam_R_w2c']
                t_match = aTemplateInfo[match.template_id]['cam_t_w2c']
                depth_ren = render(model, im_size, K_match, R_match, t_match, mode='depth')

                start_time = time.time()
                poseRefine = linemodLevelup_pybind.poseRefine()

                # make sure data type is consistent
                # have closed ICP, just point cloud conversion; you can check that refinedT[2] = one of anchor depth
                poseRefine.process(depth.astype(np.uint16), depth_ren.astype(np.uint16), K.astype(np.float32),
                                   K_match.astype(np.float32), R_match.astype(np.float32), t_match.astype(np.float32)
                                   , match.x, match.y)
                refinedR = poseRefine.getR()
                refinedT = poseRefine.getT()

                e = dict()
                e['R'] = refinedR
                e['t'] = refinedT
                e['score'] = match.similarity
                result_ests.append(e)

                render_R = refinedR
                render_t = refinedT

                elapsed_time = time.time() - start_time
                # print('residual: {}'.format(poseRefine.getResidual()))
                # print("pose refine time: {}s".format(elapsed_time))
                render_rgb_new, render_depth = render(model, im_size, render_K, render_R, render_t,
                                                      surf_color=color_list[i])
                visible_mask = render_depth < depth
                mask = render_depth > 0
                mask = mask.astype(np.uint8)
                rgb_mask = np.dstack([mask] * 3)
                render_rgb = render_rgb * (1 - rgb_mask) + render_rgb_new * rgb_mask

                draw_axis(rgb, render_R, render_t, render_K)

            result['ests'] = result_ests
            inout.save_results_sixd17(result_name, result, matching_time)

            visual = True
            # visual = False
            if visual:
                cv2.namedWindow('rgb')
                cv2.imshow('rgb', rgb)
                cv2.namedWindow('rgb_render')
                cv2.imshow('rgb_render', render_rgb)
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