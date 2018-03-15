# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates statistics of the ground truth poses, including visible fractions
# of object surfaces at the ground truth poses, 2D bounding boxes etc.

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer, visibility
from params.dataset_params import get_dataset_params

# dataset = 'hinterstoisser'
dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

dataset_part = 'train'
# dataset_part = 'test'

use_image_subset = True # Whether to use only the specified subset of images
delta = 15 # Tolerance used in the visibility test [mm]
do_vis = True # Whether to save visualizations of visibility masks

# Select data type
if dataset == 'tless':
    data_type = 'primesense'
    cam_type = 'primesense'
    model_type = 'cad'
else:
    data_type = ''
    model_type = ''
    cam_type = ''

# Load dataset parameters
dp = get_dataset_params(dataset, model_type=model_type, train_type=data_type,
                        test_type=data_type, cam_type=cam_type)
obj_ids = range(1, dp['obj_count'] + 1)

# Subset of images to be considered
if data_type == 'test' and use_image_subset:
    im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
else:
    im_ids_sets = None

if dataset_part == 'train':
    data_ids = range(1, dp['obj_count'] + 1)
    depth_mpath_key = 'train_depth_mpath'
    info_mpath_key = 'obj_info_mpath'
    gt_mpath_key = 'obj_gt_mpath'
    gt_stats_mpath_key = 'obj_gt_stats_mpath'

else: # 'test'
    data_ids = range(1, dp['scene_count'] + 1)
    depth_mpath_key = 'test_depth_mpath'
    info_mpath_key = 'scene_info_mpath'
    gt_mpath_key = 'scene_gt_mpath'
    gt_stats_mpath_key = 'scene_gt_stats_mpath'

# Path masks of the output visualizations
vis_base = '../output/vis_gt_visib_{}_delta={}/{:02d}/'
vis_mpath = vis_base + '{:' + str(dp['im_id_pad']).zfill(2) + 'd}_{:02d}.jpg'
# vis_delta_mpath = vis_base + '{:' + str(dp['im_id_pad']).zfill(2) +\
#                   'd}_{:02d}_diff_below_delta={}.jpg'

print('Loading object models...')
models = {}
for obj_id in obj_ids:
    models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

# visib_to_below_delta_fracs = []
for data_id in data_ids:
    if do_vis:
        misc.ensure_dir(os.path.dirname(
            vis_mpath.format(dataset, delta, data_id, 0, 0)))

    # Load scene info and gts
    info = inout.load_info(dp[info_mpath_key].format(data_id))
    gts = inout.load_gt(dp[gt_mpath_key].format(data_id))

    # Considered subset of images for the current scene
    if im_ids_sets is not None:
        im_ids = im_ids_sets[data_id]
    else:
        im_ids = sorted(gts.keys())

    gt_stats = {}
    for im_id in im_ids:
        print('dataset: {}, scene/obj: {}, im: {}'.format(dataset, data_id, im_id))

        K = info[im_id]['cam_K']
        depth_path = dp[depth_mpath_key].format(data_id, im_id)
        depth_im = inout.load_depth(depth_path)
        depth_im *= dp['cam']['depth_scale'] # to [mm]
        im_size = (depth_im.shape[1], depth_im.shape[0])

        gt_stats[im_id] = []
        for gt_id, gt in enumerate(gts[im_id]):
            depth_gt = renderer.render(models[gt['obj_id']], im_size, K,
                                       gt['cam_R_m2c'], gt['cam_t_m2c'],
                                       mode='depth')

            # Get distance images
            dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
            dist_im = misc.depth_im_to_dist_im(depth_im, K)

            # Estimation of visibility mask
            visib_gt = visibility.estimate_visib_mask_gt(dist_im, dist_gt, delta)

            # Visible surface fraction
            obj_mask_gt = dist_gt > 0
            px_count_valid = np.sum(dist_im[obj_mask_gt] > 0)
            px_count_visib = visib_gt.sum()
            px_count_all = obj_mask_gt.sum()
            if px_count_all > 0:
                visib_fract = px_count_visib / float(px_count_all)
            else:
                visib_fract = 0.0

            im_size = (obj_mask_gt.shape[1], obj_mask_gt.shape[0])

            # Absolute difference of the distance images
            # dist_diff = np.abs(dist_gt.astype(np.float32) -
            #                    dist_im.astype(np.float32))
            # mask_below_delta = dist_diff < delta
            # mask_below_delta *= obj_mask_gt

            # Bounding box of the object mask
            # bbox_all = [-1, -1, -1, -1]
            # if px_count_all > 0:
            #     ys, xs = obj_mask_gt.nonzero()
            #     bbox_all = misc.calc_2d_bbox(xs, ys, im_size)

            # Bounding box of the object projection
            bbox_obj = misc.calc_pose_2d_bbox(models[gt['obj_id']], im_size,
                                              K, gt['cam_R_m2c'], gt['cam_t_m2c'])

            # Bounding box of the visible surface part
            bbox_visib = [-1, -1, -1, -1]
            if px_count_visib > 0:
                ys, xs = visib_gt.nonzero()
                bbox_visib = misc.calc_2d_bbox(xs, ys, im_size)

            gt_stats[im_id].append({
                'px_count_all': int(px_count_all),
                'px_count_visib': int(px_count_visib),
                'px_count_valid': int(px_count_valid),
                'visib_fract': float(visib_fract),
                'bbox_obj': [int(e) for e in bbox_obj],
                'bbox_visib': [int(e) for e in bbox_visib]
            })

            # mask_below_delta_sum = float(mask_below_delta.sum())
            # if mask_below_delta_sum > 0:
            #     visib_to_below_delta_fracs.append({
            #         'data_id': data_id,
            #         'im_id': im_id,
            #         'gt_id': gt_id,
            #         'frac': visib_gt.sum() / float(mask_below_delta.sum())
            #     })

            if do_vis:
                # Visibility mask
                depth_im_vis = misc.norm_depth(depth_im, 0.2, 1.0)
                depth_im_vis = np.dstack([depth_im_vis] * 3)

                visib_gt_vis = visib_gt.astype(np.float)
                zero_ch = np.zeros(visib_gt_vis.shape)
                visib_gt_vis = np.dstack([zero_ch, visib_gt_vis, zero_ch])

                vis = 0.5 * depth_im_vis + 0.5 * visib_gt_vis
                vis[vis > 1] = 1
                vis_path = vis_mpath.format(
                    dataset, delta, data_id, im_id, gt_id)
                inout.save_im(vis_path, vis)

                # Mask of depth differences below delta
                # mask_below_delta_vis = np.dstack([mask_below_delta,
                #                                   zero_ch, zero_ch])
                # vis_delta = 0.5 * depth_im_vis + 0.5 * mask_below_delta_vis
                # vis_delta[vis_delta > 1] = 1
                # vis_delta_path = vis_delta_mpath.format(
                #     dataset, delta, data_id, im_id, gt_id, delta)
                # inout.save_im(vis_delta_path, vis_delta)

    res_path = dp[gt_stats_mpath_key].format(data_id, delta)
    misc.ensure_dir(os.path.dirname(res_path))
    inout.save_yaml(res_path, gt_stats)

# visib_to_below_delta_fracs = sorted(visib_to_below_delta_fracs,
#                                     key=lambda x: x['frac'], reverse=True)
# for i in range(200):
#     e = visib_to_below_delta_fracs[i]
#     print('{}: data_id: {}, im_id: {}, gt_id: {}, frac: {}'.format(
#         i, e['data_id'], e['im_id'], e['gt_id'], e['frac']
#     ))
#
# import matplotlib.pyplot as plt
# plt.plot([e['frac'] for e in visib_to_below_delta_fracs])
# plt.show()
