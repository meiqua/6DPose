import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pysixd import view_sampler, inout, misc
from pysixd.renderer import Renderer
from params.dataset_params import get_dataset_params
from os.path import join
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

obj_ids = []  # for each obj
obj_ids_curr = range(1, dp['obj_count'] + 1)
if obj_ids:
    obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

renderer = Renderer()

mode = 'test'

template_saved_to = join(dp['base_path'], 'linemod', '%s.yaml')
tempInfo_saved_to = join(dp['base_path'], 'linemod', '{:02d}_info.yaml')
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
                cv2.waitKey(1000000)

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

if mode == 'test':
    print('reading detector template & info')
    template_read_classes = []
    templateInfo = dict()
    for obj_id in obj_ids_curr:
        template_read_classes.append('{:02d}_template'.format(obj_id))
    detector.readClasses(template_read_classes, template_saved_to)

    scene_ids = [10]  # for each obj
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

        # Considered subset of images for the current scene
        if im_ids_sets is not None:
            im_ids_curr = im_ids_sets[scene_id]
        else:
            im_ids_curr = sorted(scene_info.keys())

        if im_ids:
            im_ids_curr = set(im_ids_curr).intersection(im_ids)

        for im_id in im_ids_curr:
            print('scene: {}, im: {}'.format(scene_id, im_id))

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
            output = detector.match([rgb, depth], 50, match_ids)
            elapsed_time = time.time() - start_time
            print('match time: {}s'.format(elapsed_time))
            matches = output[0]
            quantized_images = output[1]  # 2^0 --- 2^7

            most_like_match = matches[0]
            print('(x={}, y={}, float similarity={:.2f}, class_id={}, template_id={})'
                  .format(most_like_match.x, most_like_match.y, most_like_match.similarity,
                            most_like_match.class_id, most_like_match.template_id))

            template = detector.getTemplates(most_like_match.class_id, most_like_match.template_id)
            startPos = (int(most_like_match.x), int(most_like_match.y))

            # there are 4 templates, while 0-1 2-3 are same, width, height, pyramid_level
            factor1 = 2 ^ template[0].pyramid_level
            factor2 = 2 ^ template[2].pyramid_level
            print('factor1: {}, factor2: {}'.format(factor1, factor2))
            # cv2.circle(rgb, startPos, 4, (0, 0, 255), -1)
            for m in range(5): # test if top5 matches drop in right area
                startPos = (int(matches[m].x), int(matches[m].y))
                centerPos = (int(startPos[0] + im_size[0] / factor1/2), int(startPos[1] + im_size[1] / factor1/2))
                cv2.circle(rgb, centerPos, 3, (0, 0, 255), -1)
                centerPos = (int(startPos[0] + im_size[0] / factor2/2), int(startPos[1] + im_size[1] / factor2/2))
                cv2.circle(rgb, centerPos, 3, (0, 255, 0), -1)

            aTemplateInfo = inout.load_info(tempInfo_saved_to.format(scene_id))

            model = inout.load_ply(dp['model_mpath'].format(scene_id))
            render_K = aTemplateInfo[most_like_match.template_id]['cam_K']
            render_R = aTemplateInfo[most_like_match.template_id]['cam_R_w2c']
            render_t = aTemplateInfo[most_like_match.template_id]['cam_t_w2c']
            oriRBG = inout.load_im(dp['train_rgb_mpath'].format(scene_id, most_like_match.template_id))

            render_rgb, render_depth = renderer.render(model, im_size, render_K, render_R, render_t)
            visible_mask = render_depth < depth
            mask = render_depth > 0
            mask = mask.astype(np.uint8)
            rgb_mask = np.dstack([mask]*3)
            render_rgb = render_rgb*rgb_mask
            render_rgb = rgb*(1-rgb_mask) + render_rgb

            visual = True
            # visual = False
            if visual:
                # cv2.namedWindow('rgb')
                # cv2.imshow('rgb', render_rgb)
                cv2.namedWindow('depth')
                cv2.imshow('depth', rgb)
                cv2.namedWindow('oriRBG')
                cv2.imshow('oriRBG', oriRBG)
                cv2.waitKey(1000)

            gt_ids_curr = range(len(scene_gt[im_id]))
            if gt_ids:
                gt_ids_curr = set(gt_ids_curr).intersection(gt_ids)
            # for multi objs in one img
            for gt_id in gt_ids_curr:
                gt = scene_gt[im_id][gt_id]
                obj_id = gt['obj_id']

                K = scene_info[im_id]['cam_K']
                R = gt['cam_R_m2c']
                t = gt['cam_t_m2c']
                # have read rgb, depth, pose, obj_bb, obj_id here

print('end line for debug')
