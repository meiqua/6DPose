import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from pysixd import view_sampler, inout
from  pysixd.renderer import render
import linemodLevelup_pybind
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

obj_ids = []

dp = dict()
dp['im_size'] = (640, 480)
dp['model_path'] = './models/{0}/{0}.fly'
dp['depth_scale'] = 1
dp['K'] = np.asarray([572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0])

detector = linemodLevelup_pybind.Detector()

start_time = time.time()
visual = True
template_saved_to = './yaml/%s_templ.yaml'
tempInfo_saved_to = './yaml/{}_info.yaml'

ssaa_fact = 4
im_size_rgb = [int(round(x * float(ssaa_fact))) for x in dp['im_size']]
K_rgb = dp['K'] * ssaa_fact

for obj_id in obj_ids:
    templateInfo = dict()

    radii = [800, 1000]
    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)
    min_n_views = 200
    clip_near = 10  # [mm]
    clip_far = 10000  # [mm]
    ambient_weight = 0.8  # Weight of ambient light [0, 1]
    shading = 'phong'  # 'flat', 'phong'

    # Load model
    model_path = dp['model_path'].format(obj_id)
    model = inout.load_ply(model_path)

    # Load model texture
    if dp['model_texture_path']:
        model_texture_path = dp['model_texture_path'].format(obj_id)
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
            depth = render(model, dp['im_size'], dp['K'],
                           view['R'], view['t'],
                           clip_near, clip_far, mode='depth')

            # Convert depth so it is in the same units as the real test images
            depth /= dp['depth_scale']
            depth = depth.astype(np.uint16)

            # Render RGB image
            rgb = render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                         clip_near, clip_far, texture=model_texture,
                         ambient_weight=ambient_weight, shading=shading,
                         mode='rgb')
            rgb = cv2.resize(rgb, dp['im_size'], interpolation=cv2.INTER_AREA)

            K = dp['K']
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
                cv2.namedWindow('depth')
                cv2.imshow('depth', depth)
                cv2.namedWindow('mask')
                cv2.imshow('mask', mask)
                cv2.waitKey(1000)

            success = detector.addTemplate([rgb, depth], '{}_templ'.format(obj_id), mask)
            print('success {}'.format(success))
            del rgb, depth, mask

            if success != -1:
                templateInfo[success] = aTemplateInfo

    inout.save_info(tempInfo_saved_to.format(obj_id), templateInfo)

detector.writeClasses(template_saved_to)
elapsed_time = time.time() - start_time
print('train time: {}\n'.format(elapsed_time))