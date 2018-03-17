import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import view_sampler, inout, misc
from pysixd.renderer import Renderer
from params.dataset_params import get_dataset_params
from os.path import join

dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

dp = get_dataset_params(dataset)

# train
# test
# render
mode = 'render'


if mode == 'test':
    scene_ids = []  # for each obj
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
            depth = depth.astype(np.float32)  # [mm]
            depth *= dp['cam']['depth_scale']  # to [mm]

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

elif mode == 'train':
    obj_ids = []  # for each obj
    im_ids = []  # obj's img

    obj_ids_curr = range(1, dp['obj_count'] + 1)
    if obj_ids:
        obj_ids_curr = set(obj_ids_curr).intersection(obj_ids)

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
            depth = depth.astype(np.float32)  # [mm]
            depth *= dp['cam']['depth_scale']  # to [mm]

            # during training, there's only one obj
            gt = scene_gt[im_id][0]
            obj_id = gt['obj_id']

            K = scene_info[im_id]['cam_K']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            # have read rgb, depth, pose, obj_bb, obj_id here

elif mode == 'render':
    renderer = Renderer()
    radii = [400]  # Radii of the view sphere [mm]
    azimuth_range = (0, 2 * math.pi)
    elev_range = (0, 0.5 * math.pi)

    p = dict()
    p['name'] = 'customDataset'
    p['obj_count'] = 2
    p['scene_count'] = 2
    p['train_im_size'] = (640, 480)
    p['test_im_size'] = (640, 480)
    p['base_path'] = join('/home/meiqua/6DPose/public/datasets/', p['name'])
    misc.ensure_dir(p['base_path'])

    p['model_mpath'] = join(p['base_path'], 'models', 'obj_{:02d}.ply')
    misc.ensure_dir(os.path.dirname(p['model_mpath']))
    p['model_texture_mpath'] = join(p['base_path'], 'models', 'obj_{:02d}.png')

    p['cam_params_path'] = join(p['base_path'], 'camera.yml')
    p['cam'] = inout.load_cam_params(p['cam_params_path'])
    # Minimum required number of views on the whole view sphere. The final number of
    # views depends on the sampling method.
    min_n_views = 100

    clip_near = 10  # [mm]
    clip_far = 10000  # [mm]
    ambient_weight = 0.8  # Weight of ambient light [0, 1]
    shading = 'phong'  # 'flat', 'phong'

    # Super-sampling anti-aliasing (SSAA)
    # https://github.com/vispy/vispy/wiki/Tech.-Antialiasing
    # The RGB image is rendered at ssaa_fact times higher resolution and then
    # down-sampled to the required resolution.
    ssaa_fact = 4

    # Output path masks
    out_rgb_mpath = join(p['base_path'], 'train', '{:02d}/rgb/{:04d}.png')
    out_depth_mpath = join(p['base_path'], 'train', '{:02d}/depth/{:04d}.png')
    out_obj_info_path = join(p['base_path'], 'train', '{:02d}/info.yml')
    out_obj_gt_path = join(p['base_path'], 'train', '{:02d}/gt.yml')
    out_views_vis_mpath = join(p['base_path'], 'views_radius={}.ply')

    # Prepare output folder
    # misc.ensure_dir(os.path.dirname(out_obj_info_path))

    # Image size and K for SSAA
    im_size_rgb = [int(round(x * float(ssaa_fact))) for x in p['cam']['im_size']]
    K_rgb = p['cam']['K'] * ssaa_fact

    obj_ids = range(1, p['obj_count']+1)
    for obj_id in obj_ids:
        # Prepare folders
        misc.ensure_dir(os.path.dirname(out_rgb_mpath.format(obj_id, 0)))
        misc.ensure_dir(os.path.dirname(out_depth_mpath.format(obj_id, 0)))

        # Load model
        model_path = p['model_mpath'].format(obj_id)
        model = inout.load_ply(model_path)

        # Load model texture
        if p['model_texture_mpath']:
            model_texture_path = p['model_texture_mpath'].format(obj_id)
            model_texture = inout.load_im(model_texture_path)
        else:
            model_texture = None

        obj_info = {}
        obj_gt = {}
        im_id = 0
        for radius in radii:
            # Sample views
            views, views_level = view_sampler.sample_views(min_n_views, radius,
                                                           azimuth_range, elev_range)
            print('Sampled views: ' + str(len(views)))
            # view_sampler.save_vis(out_views_vis_mpath.format(str(radius)),
            #                       views, views_level)

            # Render the object model from all the views
            for view_id, view in enumerate(views):
                if view_id % 10 == 0:
                    print('obj,radius,view: ' + str(obj_id) +
                          ',' + str(radius) + ',' + str(view_id))

                # Render depth image
                depth = renderer.render(model, p['cam']['im_size'], p['cam']['K'],
                                        view['R'], view['t'],
                                        clip_near, clip_far, mode='depth')

                # Convert depth so it is in the same units as the real test images
                depth /= p['cam']['depth_scale']

                # Render RGB image
                rgb = renderer.render(model, im_size_rgb, K_rgb, view['R'], view['t'],
                                      clip_near, clip_far, texture=model_texture,
                                      ambient_weight=ambient_weight, shading=shading,
                                      mode='rgb')

                # The OpenCV function was used for rendering of the training images
                # provided for the SIXD Challenge 2017.
                rgb = cv2.resize(rgb, p['cam']['im_size'], interpolation=cv2.INTER_AREA)
                # rgb = scipy.misc.imresize(rgb, par['cam']['im_size'][::-1], 'bicubic')

                # Save the rendered images
                inout.save_im(out_rgb_mpath.format(obj_id, im_id), rgb)
                inout.save_depth(out_depth_mpath.format(obj_id, im_id), depth)

                # Get 2D bounding box of the object model at the ground truth pose
                ys, xs = np.nonzero(depth > 0)
                obj_bb = misc.calc_2d_bbox(xs, ys, p['cam']['im_size'])

                obj_info[im_id] = {
                    'cam_K': p['cam']['K'].flatten().tolist(),
                    'view_level': int(views_level[view_id]),
                    # 'sphere_radius': float(radius)
                }

                obj_gt[im_id] = [{
                    'cam_R_m2c': view['R'].flatten().tolist(),
                    'cam_t_m2c': view['t'].flatten().tolist(),
                    'obj_bb': [int(x) for x in obj_bb],
                    'obj_id': int(obj_id)
                }]

                im_id += 1

        # Save metadata
        inout.save_yaml(out_obj_info_path.format(obj_id), obj_info)
        inout.save_yaml(out_obj_gt_path.format(obj_id), obj_gt)

print("end of line for break points")