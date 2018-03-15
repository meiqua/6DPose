# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Visualizes 6D object pose estimates from files in the SIXD format.

import os
from os.path import join as pjoin
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout, misc, renderer
from params.dataset_params import get_dataset_params

#-------------------------------------------------------------------------------
result_base = '../public/sixd_results/'

result_paths = [
    # pjoin(result_base, 'hodan-iros15_hinterstoisser'),
    pjoin(result_base, 'hodan-iros15_tless_primesense'),
]

# Other paths
#-------------------------------------------------------------------------------
# Mask of path to the output visualization file
vis_path = '{result_path}/../../vis/{result_name}/{scene_id:02d}/'
vis_rgb_mpath = vis_path + '{im_id:05d}_{obj_id:02d}.jpg'
vis_depth_mpath = vis_path + '{im_id:05d}_{obj_id:02d}_depth_diff.jpg'

# Parameters
#-------------------------------------------------------------------------------
# Top N pose estimates (with the highest score) to be displayed for each
# object in each image
n_top = -1 # 0 = all estimates, -1 = given by the number of GT poses

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

# Object colors (used if vis_orig_colors == False)
colors = inout.load_yaml('../data/colors.yml')

assert(vis_rgb or vis_depth)

# Visualization
#-------------------------------------------------------------------------------
for result_path in result_paths:
    print('Processing: ' + result_path)

    result_name = os.path.basename(result_path)
    info = result_name.split('_')
    method = info[0]
    dataset = info[1]
    test_type = info[2] if len(info) > 2 else ''

    # Select data type
    if dataset == 'tless':
        cam_type = test_type
        model_type = 'cad_subdivided'
    else:
        model_type = ''
        cam_type = ''

    # Load dataset parameters
    dp = get_dataset_params(dataset, model_type=model_type, test_type=test_type,
                            cam_type=cam_type)

    # Load object models
    print('Loading object models...')
    models = {}
    for obj_id in range(1, dp['obj_count'] + 1):
        models[obj_id] = inout.load_ply(dp['model_mpath'].format(obj_id))

    # Directories with results for individual scenes
    scene_dirs = sorted([d for d in glob.glob(os.path.join(result_path, '*'))
                         if os.path.isdir(d)])

    for scene_dir in scene_dirs:
        scene_id = int(os.path.basename(scene_dir))

        # Load info and GT poses for the current scene
        scene_info = inout.load_info(dp['scene_info_mpath'].format(scene_id))
        scene_gt = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

        res_paths = sorted(
            glob.glob(os.path.join(scene_dir, '*.yml')) +
            glob.glob(os.path.join(scene_dir, '*.yaml'))
        )
        im_id = -1
        depth_im = None
        for res_id, res_path in enumerate(res_paths):

            # Parse image ID and object ID from the file name
            filename = os.path.basename(res_path).split('.')[0]
            im_id_prev = im_id
            im_id, obj_id = map(int, filename.split('_'))

            if res_id % 10 == 0:
                print('Processing: {}, {}, {}, {}, {}, {}'.format(
                    method, dataset, test_type, scene_id, im_id, obj_id))

            # Colors
            if vis_orig_color:
                color = (1, 1, 1)
            else:
                color = tuple(colors[(obj_id - 1) % len(colors)])
            color_uint8 = tuple([int(255 * c) for c in color])

            # Load the RGB-D image
            im_size = None
            if vis_rgb:
                rgb_path = dp['test_rgb_mpath'].format(scene_id, im_id)
                rgb = inout.load_im(rgb_path)
                ren_rgb = np.zeros(rgb.shape, np.float32)
                ren_rgb_info = np.zeros(rgb.shape, np.uint8)
                im_size = (rgb.shape[1], rgb.shape[0])

            if vis_depth:
                depth_path = dp['test_depth_mpath'].format(scene_id, im_id)
                depth = inout.load_depth(depth_path)
                depth *= dp['cam']['depth_scale'] # to [mm]
                if im_size:
                    assert(im_size == (depth.shape[1], depth.shape[0]))
                else:
                    im_size = (depth.shape[1], depth.shape[0])

            if vis_depth or (vis_rgb and vis_rgb_resolve_visib):
                ren_depth = np.zeros((im_size[1], im_size[0]), np.float32)

            # Load camera matrix
            K = scene_info[im_id]['cam_K']

            # Load pose estimates
            res = inout.load_results_sixd17(res_path)
            ests = res['ests']

            # Sort the estimates by score (in descending order)
            ests_sorted = sorted(enumerate(ests), key=lambda x: x[1]['score'],
                                 reverse=True)

            # Select the required number of top estimated poses
            if n_top == 0: # All estimates are considered
                n_top_curr = None
            elif n_top == -1: # Given by the number of GT poses
                n_gt = sum([gt['obj_id'] == obj_id for gt in scene_gt[im_id]])
                n_top_curr = n_gt
            else:
                n_top_curr = n_top
            ests_sorted = ests_sorted[slice(0, n_top_curr)]

            for est_id, est in ests_sorted:
                est_errs = []
                R_e = est['R']
                t_e = est['t']
                score = est['score']

                # Rendering
                model = models[obj_id]
                if vis_rgb:
                    if vis_orig_color:
                        m_rgb = renderer.render(
                            model, im_size, K, R_e, t_e, mode='rgb')
                    else:
                        m_rgb = renderer.render(
                            model, im_size, K, R_e, t_e, mode='rgb',
                            surf_color=color)

                if vis_depth or (vis_rgb and vis_rgb_resolve_visib):
                    m_depth = renderer.render(
                        model, im_size, K, R_e, t_e, mode='depth')

                    # Get mask of the surface parts that are closer than the
                    # surfaces rendered before
                    visible_mask = np.logical_or(ren_depth == 0,
                                                 m_depth < ren_depth)
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

                            txt = '{},{:.2f}'.format(obj_id, score)
                            txt_offset = 0
                            # txt_offset = 5
                            p_id = np.argmin(ys)
                            tex_loc = (xs[p_id], ys[p_id] - 5)
                            # tex_loc = (bbox[0], bbox[1])
                            cv2.putText(ren_rgb_info, txt, tex_loc,
                                        cv2.FONT_HERSHEY_PLAIN, 0.9,
                                        color_uint8, 1)

            # Save RGB visualization
            if vis_rgb:
                vis_im_rgb = 0.5 * rgb.astype(np.float32) + \
                             0.5 * ren_rgb + \
                             1.0 * ren_rgb_info
                vis_im_rgb[vis_im_rgb > 255] = 255
                vis_rgb_path = vis_rgb_mpath.format(
                    result_path=result_path, result_name=result_name,
                    scene_id=scene_id, im_id=im_id, obj_id=obj_id)
                misc.ensure_dir(os.path.dirname(vis_rgb_path))
                inout.save_im(vis_rgb_path, vis_im_rgb.astype(np.uint8))

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
                vis_depth_path = vis_depth_mpath.format(
                    result_path=result_path, result_name=result_name,
                    scene_id=scene_id, im_id=im_id, obj_id=obj_id)
                if not vis_rgb:
                    misc.ensure_dir(os.path.dirname(vis_depth_path))
                plt.savefig(vis_depth_path, pad=0, bbox_inches='tight')
                plt.close()

    print('')
print('Done.')
