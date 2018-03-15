# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# A script to render 3D object models into the test images. The models are
# rendered at the ground truth 6D poses that are provided with the test images.
# The visualizations are saved into the folder specified by "output_dir".

from pytless import inout, renderer, misc
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

scene_ids = range(1, 21)
device = 'primesense' # options: 'primesense', 'kinect', 'canon'
model_type = 'cad' # options: 'cad', 'reconst'
im_step = 100 # Consider every im_step-th image

# Path to the T-LESS dataset
data_path = '/local/datasets/sixd/t-less/t-less_v2'

# Path to the folder in which the images produced by this script will be saved
output_dir = os.path.join(data_path, 'output_check_poses_test_imgs')

# Paths to the elements of the T-LESS dataset
model_path_mask = os.path.join(data_path, 'models_' + model_type, 'obj_{:02d}.ply')
scene_info_path_mask = os.path.join(data_path, 'test_{}', '{:02d}', 'info.yml')
scene_gt_path_mask = os.path.join(data_path, 'test_{}', '{:02d}', 'gt.yml')
rgb_path_mask = os.path.join(data_path, 'test_{}', '{:02d}', 'rgb', '{:04d}.{}')
depth_path_mask = os.path.join(data_path, 'test_{}', '{:02d}', 'depth', '{:04d}.png')
rgb_ext = {'primesense': 'png', 'kinect': 'png', 'canon': 'jpg'}
obj_colors_path = os.path.join('data', 'obj_rgb.txt')
vis_rgb_path_mask = os.path.join(output_dir, '{:02d}_{}_{}_{:04d}_rgb.png')
vis_depth_path_mask = os.path.join(output_dir, '{:02d}_{}_{}_{:04d}_depth_diff.png')

misc.ensure_dir(output_dir)
obj_colors = inout.load_colors(obj_colors_path)

plt.ioff() # Turn interactive plotting off

for scene_id in scene_ids:

    # Load info about the test images (including camera parameters etc.)
    scene_info_path = scene_info_path_mask.format(device, scene_id)
    scene_info = inout.load_info(scene_info_path)

    scene_gt_path = scene_gt_path_mask.format(device, scene_id)
    scene_gt = inout.load_gt(scene_gt_path)

    # Load models of objects present in the scene
    scene_obj_ids = set()
    for gt in scene_gt[0]:
        scene_obj_ids.add(gt['obj_id'])
    models = {}
    for scene_obj_id in scene_obj_ids:
        model_path = model_path_mask.format(scene_obj_id)
        models[scene_obj_id] = inout.load_ply(model_path)

    for im_id, im_info in scene_info.items():
        if im_id % im_step != 0:
            continue
        print('scene: ' + str(scene_id) + ', device: ' + device + ', im_id: ' + str(im_id))

        # Get intrinsic camera parameters
        K = im_info['cam_K']

        # Visualization #1
        #-----------------------------------------------------------------------
        # Load RGB image
        rgb_path = rgb_path_mask.format(device, scene_id, im_id, rgb_ext[device])
        rgb = scipy.misc.imread(rgb_path)

        im_size = (rgb.shape[1], rgb.shape[0])
        vis_rgb = np.zeros(rgb.shape, np.float)
        for gt in scene_gt[im_id]:
            model = models[gt['obj_id']]
            R = gt['cam_R_m2c']
            t = gt['cam_t_m2c']
            surf_color = obj_colors[gt['obj_id'] - 1]

            ren_rgb = renderer.render(model, im_size, K, R, t,
                                      surf_color=surf_color, mode='rgb')

            # Draw the bounding box of the object
            ren_rgb = misc.draw_rect(ren_rgb, gt['obj_bb'])

            vis_rgb += 0.7 * ren_rgb.astype(np.float)

        # Save the visualization
        vis_rgb = 0.6 * vis_rgb + 0.4 * rgb
        vis_rgb[vis_rgb > 255] = 255
        vis_rgb_path = vis_rgb_path_mask.format(scene_id, device, model_type, im_id)
        scipy.misc.imsave(vis_rgb_path, vis_rgb.astype(np.uint8))

        # Visualization #2
        #-----------------------------------------------------------------------
        if device != 'canon':
            # Load depth image
            depth_path = depth_path_mask.format(device, scene_id, im_id, rgb_ext[device])
            depth = scipy.misc.imread(depth_path)  # Unit: 0.1 mm
            depth = depth.astype(np.float) * 0.1  # Convert to mm

            # Render the objects at the ground truth poses
            im_size = (depth.shape[1], depth.shape[0])
            ren_depth = np.zeros(depth.shape, np.float)
            for gt in scene_gt[im_id]:
                model = models[gt['obj_id']]
                R = gt['cam_R_m2c']
                t = gt['cam_t_m2c']

                # Render the current object
                ren_depth_obj = renderer.render(model, im_size, K, R, t, mode='depth')

                # Add to the final depth map only the parts of the surface that
                # are closer than the surfaces rendered before
                visible_mask = np.logical_or(ren_depth == 0, ren_depth_obj < ren_depth)
                mask = np.logical_and(ren_depth_obj != 0, visible_mask)
                ren_depth[mask] = ren_depth_obj[mask].astype(np.float)

            # Calculate the depth difference at pixels where both depth maps
            # are valid
            valid_mask = (depth > 0) * (ren_depth > 0)
            depth_diff = valid_mask * (depth - ren_depth.astype(np.float))

            # Save the visualization
            vis_depth_path = vis_depth_path_mask.format(scene_id, device,
                                                        model_type, im_id)
            plt.matshow(depth_diff)
            plt.title('captured - rendered depth [mm]')
            plt.colorbar()
            plt.savefig(vis_depth_path, pad=0)
            plt.close()
