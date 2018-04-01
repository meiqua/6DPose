import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pysixd import view_sampler, inout, misc
from  pysixd.renderer import render
from params.dataset_params import get_dataset_params
from os.path import join

import cxxlinemod_pybind

# import linemodLevelup_pybind

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

dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

# set ./params/dataset_params common_base_path correctly
dp = get_dataset_params(dataset)
detector = cv2.linemod.getDefaultLINEMOD()
# detector = linemodLevelup_pybind.Detector()
obj_ids = []  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

# renderer = Renderer()

mode = 'test'

# template_saved_to = join(dp['base_path'], 'linemod', '%s.yaml')
# tempInfo_saved_to = join(dp['base_path'], 'linemod', '{:02d}_info.yaml')
template_saved_to = join(dp['base_path'], 'linemod_render', '%s.yaml')
tempInfo_saved_to = join(dp['base_path'], 'linemod_render', '{:02d}_info.yaml')
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

            # convert to float32 will fail, after a painful try under c++ T_T
            depth = depth.astype(np.uint16)  # [mm]
            # depth *= dp['cam']['depth_scale']  # to [mm]

            # depth /= 1000.0  # [m]
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
                cv2.waitKey(1000)

            # test what will happen if addTemplate fails
            # no template will be added, rather than a empty template
            # if im_id % 10 == 0:
            #     depth = depth.astype(np.float32)

            success = detector.addTemplate([rgb, depth], '{:02d}_template'.format(obj_id), mask)
            print('success {}'.format(success))

            if success[0] != -1:
                templateInfo[success[0]] = aTemplateInfo

        inout.save_info(tempInfo_saved_to.format(obj_id), templateInfo)

    detector.writeClasses(template_saved_to)
    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'render_train':
    start_time = time.time()
    visual = True
    misc.ensure_dir(os.path.dirname(template_saved_to))

    # Super-sampling anti-aliasing (SSAA)
    # https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
    # The RGB image is rendered at ssaa_fact times higher resolution and then
    # down-sampled to the required resolution.
    ssaa_fact = 4
    im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['cam']['im_size']]
    K_rgb = dp['cam']['K'] * ssaa_fact

    for obj_id in obj_ids_curr:
        templateInfo = dict()

        radii = [600, 700, 800, 900, 1000]
        azimuth_range = (0, 2 * math.pi)
        elev_range = (0, 0.5 * math.pi)
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

        im_id = 0
        for radius in radii:
            # Sample views
            views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                           azimuth_range, elev_range)
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

                aTemplateInfo = dict()
                aTemplateInfo['cam_K'] = K
                aTemplateInfo['cam_R_w2c'] = R
                aTemplateInfo['cam_t_w2c'] = t

                mask = (depth > 0).astype(np.uint8) * 255

                visual = False
                if visual:
                    cv2.namedWindow('rgb')
                    cv2.imshow('rgb', rgb)
                    cv2.namedWindow('depth')
                    cv2.imshow('depth', depth)
                    cv2.namedWindow('mask')
                    cv2.imshow('mask', mask)
                    cv2.waitKey(1000)

                success = detector.addTemplate([rgb, depth], '{:02d}_template'.format(obj_id), mask)
                print('success {}'.format(success))
                del rgb, depth, mask

                if success[0] != -1:
                    templateInfo[success[0]] = aTemplateInfo

        inout.save_info(tempInfo_saved_to.format(obj_id), templateInfo)

    detector.writeClasses(template_saved_to)
    elapsed_time = time.time() - start_time
    print('train time: {}\n'.format(elapsed_time))

if mode == 'test':
    print('reading detector template & info')
    template_read_classes = []
    templateInfo = dict()
    for obj_id in obj_ids_curr:
        template_read_classes.append('{:02d}_template'.format(obj_id))
    detector.readClasses(template_read_classes, template_saved_to)

    scene_ids = [6]  # for each obj
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
            depth = depth.astype(np.uint16)  # [mm]
            # depth *= dp['cam']['depth_scale']  # to [mm]
            im_size = (depth.shape[1], depth.shape[0])
            start_time = time.time()
            match_ids = list()
            match_ids.append('{:02d}_template'.format(scene_id))

            # only search for one obj
            output = detector.match([rgb, depth], 90, match_ids)
            elapsed_time = time.time() - start_time

            matches = output[0]
            quantized_images = output[1]  # 2^0 --- 2^7
            print('match time: {}s, {} matches'.format(elapsed_time, len(matches)))
            most_like_match = matches[0]
            print('(x={}, y={}, float similarity={:.2f}, class_id={}, template_id={})'
                  .format(most_like_match.x, most_like_match.y, most_like_match.similarity,
                            most_like_match.class_id, most_like_match.template_id))

            template = detector.getTemplates(most_like_match.class_id, most_like_match.template_id)
            startPos = (int(most_like_match.x), int(most_like_match.y))

            # for m in range(5): # test if top5 matches drop in right area
            #     startPos = (int(matches[m].x), int(matches[m].y))
            #     template = detector.getTemplates(matches[m].class_id, matches[m].template_id)
            #     factor1 = 2 ^ template[0].pyramid_level
            #
            #     centerPos = (int(startPos[0] + template[0].width/2 ), int(startPos[1] + template[0].height/2))
            #     tempR = max(template[0].width/2, template[0].height/2)
            #     cv2.circle(rgb, centerPos, int(tempR), (0, 0, 255), 2)

            render_K = aTemplateInfo[most_like_match.template_id]['cam_K']
            render_R = aTemplateInfo[most_like_match.template_id]['cam_R_w2c']
            render_t = aTemplateInfo[most_like_match.template_id]['cam_t_w2c']

            for i in range(1):
                match = matches[i]
                template = detector.getTemplates(match.class_id, match.template_id)
                startPos = (int(match.x), int(match.y))
                K_match = aTemplateInfo[match.template_id]['cam_K']
                R_match = aTemplateInfo[match.template_id]['cam_R_w2c']
                t_match = aTemplateInfo[match.template_id]['cam_t_w2c']
                depth_ren = render(model, im_size, K_match, R_match, t_match, mode='depth')

                start_time = time.time()
                poseRefine = cxxlinemod_pybind.poseRefine()
                # make sure data type is consistent
                poseRefine.process(depth.astype(np.uint16), depth_ren.astype(np.uint16), K.astype(np.float32),
                                   K_match.astype(np.float32), R_match.astype(np.float32), t_match.astype(np.float32)
                                   , match.x, match.y)
                refinedR = poseRefine.getR()
                refinedT = poseRefine.getT()

                render_R = refinedR
                render_t = refinedT

                print('residual: {}'.format(poseRefine.getResidual()))

                elapsed_time = time.time() - start_time
                print("pose refine time: {}s".format(elapsed_time))

            render_rgb, render_depth = render(model, im_size, render_K, render_R, render_t, surf_color=[0, 1, 0])
            visible_mask = render_depth < depth
            mask = render_depth > 0
            mask = mask.astype(np.uint8)
            rgb_mask = np.dstack([mask]*3)
            render_rgb = render_rgb*rgb_mask
            render_rgb = rgb*(1-rgb_mask) + render_rgb

            draw_axis(rgb, render_R, render_t, render_K)
            # draw_axis(render_rgb, render_R, render_t, render_K)

            visual = True
            # visual = False
            if visual:
                cv2.namedWindow('rgb_render')
                cv2.imshow('rgb_render', render_rgb)
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
