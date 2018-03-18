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

mode = 'train'

if mode == 'train':
    start_time = time.time()
    # im_ids = list(range(1, 1000, 10))  # obj's img
    im_ids = []
    visual = True
    templateInfo = dict()
    template_saved_to = join(dp['base_path'], 'linemod', '%s.yaml')
    tempInfo_saved_to = join(dp['base_path'], 'linemod', '{:02d}_info.yaml')
    misc.ensure_dir(os.path.dirname(template_saved_to))

    for obj_id in obj_ids_curr:
        scene_info = inout.load_info(dp['obj_info_mpath'].format(obj_id))
        scene_gt = inout.load_gt(dp['obj_gt_mpath'].format(obj_id))

        im_ids_curr = sorted(scene_info.keys())

        if im_ids:
            im_ids_curr = set(im_ids_curr).intersection(im_ids)

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

            visual = False
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

print('end line for debug')
