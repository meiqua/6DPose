# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Visualizes the object models at the ground truth poses.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer
from params.dataset_params import get_dataset_params

dataset = 'hinterstoisser'
# dataset = 'tless'
# dataset = 'tudlight'
# dataset = 'rutgers'
# dataset = 'tejani'
# dataset = 'doumanoglou'
# dataset = 'toyotalight'

# Dataset parameters
dp = get_dataset_params(dataset)

# Select IDs of scenes, images and GT poses to be processed.
# Empty list [] means that all IDs will be used.
scene_ids = []
im_ids = []
gt_ids = []

# Indicates whether to render RGB image
vis_rgb = True

# Indicates whether to resolve visibility in the rendered RGB image (using
# depth renderings). If True, only the part of object surface, which is not
# occluded by any other modeled object, is visible. If False, RGB renderings
# of individual objects are blended together.
vis_rgb_resolve_visib = True

# Indicates whether to render depth image
vis_depth = False

# If to use the original model color
vis_orig_color = False

# Define new object colors (used if vis_orig_colors == False)
colors = inout.load_yaml('../data/colors.yml')

# Path masks for output images
vis_rgb_mpath = '../output/vis_gt_poses_{}/{:02d}/{:04d}.jpg'
vis_depth_mpath = '../output/vis_gt_poses_{}/{:02d}/{:04d}_depth_diff.jpg'

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
    misc.ensure_dir(os.path.dirname(vis_rgb_mpath.format(dataset, scene_id, 0)))

    # Load scene info and gt poses
    scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
    scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

    # Load models of objects that appear in the current scene
    obj_ids = set([gt['obj_id'] for gts in scene_gt.values() for gt in gts])
    models = {}
    for obj_id in obj_ids:
        models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

    # Considered subset of images for the current scene
    if im_ids_sets is not None:
        im_ids_curr = im_ids_sets[scene_id]
    else:
        im_ids_curr = sorted(scene_info.keys())

    if im_ids:
        im_ids_curr = set(im_ids_curr).intersection(im_ids)

    # Visualize GT poses in the selected images
    for im_id in im_ids_curr:
        print('scene: {}, im: {}'.format(scene_id, im_id))

        # Load the images
        rgb = inout.load_im(dp['test_rgb_mpath'].format(scene_id, im_id))
        depth = inout.load_depth(dp['test_depth_mpath'].format(scene_id, im_id))
        depth = depth.astype(np.float32) # [mm]
        depth *= dp['cam']['depth_scale'] # to [mm]

        # Render the objects at the ground truth poses
        im_size = (depth.shape[1], depth.shape[0])
        ren_rgb = np.zeros(rgb.shape, np.float32)
        ren_rgb_info = np.zeros(rgb.shape, np.uint8)
        ren_depth = np.zeros(depth.shape, np.float32)

        gt_ids_curr = range(len(scene_gt[im_id]))
        if gt_ids:
            gt_ids_curr = set(gt_ids_curr).intersection(gt_ids)
        for gt_id in gt_ids_curr:
            gt = scene_gt[im_id][gt_id]
            obj_id = gt['obj_id']
            if vis_orig_color:
                color = (1, 1, 1)
            else:
                color = tuple(colors[(obj_id - 1) % len(colors)])
            color_uint8 = tuple([int(255 * c) for c in color])

            model = models[gt['obj_id']]
            K = scene_info[im_id]['cam_K']
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']

            # Rendering
            if vis_rgb:
                if vis_orig_color:
                    m_rgb = renderer.render(model, im_size, K, R, t, mode='rgb')
                else:
                    m_rgb = renderer.render(model, im_size, K, R, t, mode='rgb',
                                            surf_color=color)

            if vis_depth or (vis_rgb and vis_rgb_resolve_visib):
                m_depth = renderer.render(model, im_size, K, R, t, mode='depth')

                # Get mask of the surface parts that are closer than the
                # surfaces rendered before
                visible_mask = np.logical_or(ren_depth == 0, m_depth < ren_depth)
                mask = np.logical_and(m_depth != 0, visible_mask)

                ren_depth[mask] = m_depth[mask].astype(ren_depth.dtype)

            # Combine the RGB renderings
            if vis_rgb:
                if vis_rgb_resolve_visib:
                    ren_rgb[mask] = m_rgb[mask].astype(ren_rgb.dtype)
                else:
                    ren_rgb += m_rgb.astype(ren_rgb.dtype)

                # Draw 2D bounding box and info
                if True:
                    obj_mask = np.sum(m_rgb > 0, axis=2)
                    ys, xs = obj_mask.nonzero()
                    if len(ys):
                        # bbox = misc.calc_2d_bbox(xs, ys, im_size)
                        # im_size = (obj_mask.shape[1], obj_mask.shape[0])
                        # ren_rgb_bbs = misc.draw_rect(ren_rgb_bbs, bbox, color_uint8)

                        txt = '{},{}'.format(obj_id, gt_id)
                        txt_offset = 0
                        # txt_offset = 5
                        p_id = np.argmin(ys)
                        tex_loc = (xs[p_id], ys[p_id] - 5)
                        # tex_loc = (bbox[0], bbox[1])
                        cv2.putText(ren_rgb_info, txt, tex_loc,
                                    cv2.FONT_HERSHEY_PLAIN, 0.9, color_uint8, 1)

        # Save RGB visualization
        if vis_rgb:
            vis_im_rgb = 0.5 * rgb.astype(np.float32) +\
                         0.5 * ren_rgb + \
                         1.0 * ren_rgb_info
            vis_im_rgb[vis_im_rgb > 255] = 255
            inout.save_im(vis_rgb_mpath.format(dataset, scene_id, im_id),
                          vis_im_rgb.astype(np.uint8))

        # Save image of depth differences
        if vis_depth:
            # Calculate the depth difference at pixels where both depth maps
            # are valid
            valid_mask = (depth > 0) * (ren_depth > 0)
            depth_diff = valid_mask * (depth - ren_depth.astype(np.float32))

            f, ax = plt.subplots(1, 1)
            cax = ax.matshow(depth_diff)
            ax.axis('off')
            ax.set_title('measured - GT depth [mm]')
            f.colorbar(cax, fraction=0.03, pad=0.01)
            f.tight_layout(pad=0)
            plt.savefig(vis_depth_mpath.format(dataset, scene_id, im_id), pad=0,
                        bbox_inches='tight')
            plt.close()
