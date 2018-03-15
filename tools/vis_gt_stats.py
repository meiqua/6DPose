# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Plots statistics of the ground truth poses.

import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pysixd import inout
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

# Load dataset parameters
dp = get_dataset_params(dataset)

if dataset_part == 'train':
    data_ids = range(1, dp['obj_count'] + 1)
    gt_mpath_key = 'obj_gt_mpath'
    gt_stats_mpath_key = 'obj_gt_stats_mpath'

else: # 'test'
    data_ids = range(1, dp['scene_count'] + 1)
    gt_mpath_key = 'scene_gt_mpath'
    gt_stats_mpath_key = 'scene_gt_stats_mpath'

# Subset of images to be considered
if dataset_part == 'test' and use_image_subset:
    im_ids_sets = inout.load_yaml(dp['test_set_fpath'])
else:
    im_ids_sets = None

# Load the GT statistics
gt_stats = []
for data_id in data_ids:
    print('Loading GT stats: {}, {}'.format(dataset, data_id))
    gts = inout.load_gt(dp[gt_mpath_key].format(data_id))
    gt_stats_curr = inout.load_yaml(
        dp[gt_stats_mpath_key].format(data_id, delta))

    # Considered subset of images for the current scene
    if im_ids_sets is not None:
        im_ids_curr = im_ids_sets[data_id]
    else:
        im_ids_curr = sorted(gt_stats_curr.keys())

    for im_id in im_ids_curr:
        gt_stats_im = gt_stats_curr[im_id]
        for gt_id, p in enumerate(gt_stats_im):
            p['data_id'] = data_id
            p['im_id'] = im_id
            p['gt_id'] = gt_id
            p['obj_id'] = gts[im_id][gt_id]['obj_id']
            gt_stats.append(p)

print('GT count: {}'.format(len(gt_stats)))

# Collect the data
px_count_all = [p['px_count_all'] for p in gt_stats]
px_count_valid = [p['px_count_valid'] for p in gt_stats]
px_count_visib = [p['px_count_visib'] for p in gt_stats]
visib_fract = [p['visib_fract'] for p in gt_stats]
bbox_all_x = [p['bbox_all'][0] for p in gt_stats]
bbox_all_y = [p['bbox_all'][1] for p in gt_stats]
bbox_all_w = [p['bbox_all'][2] for p in gt_stats]
bbox_all_h = [p['bbox_all'][3] for p in gt_stats]
bbox_visib_x = [p['bbox_visib'][0] for p in gt_stats]
bbox_visib_y = [p['bbox_visib'][1] for p in gt_stats]
bbox_visib_w = [p['bbox_visib'][2] for p in gt_stats]
bbox_visib_h = [p['bbox_visib'][3] for p in gt_stats]

f, axs = plt.subplots(2, 2)
f.canvas.set_window_title(dataset)

axs[0, 0].hist([px_count_all, px_count_valid, px_count_visib],
            bins=20, range=(min(px_count_visib), max(px_count_all)))
axs[0, 0].legend([
    'All object mask pixels',
    'Valid object mask pixels',
    'Visible object mask pixels'
])

axs[0, 1].hist(visib_fract, bins=50, range=(0.0, 1.0))
axs[0, 1].set_xlabel('Visible fraction')

axs[1, 0].hist([bbox_all_x, bbox_all_y, bbox_visib_x, bbox_visib_y], bins=20)
axs[1, 0].legend([
    'Bbox all - x',
    'Bbox all - y',
    'Bbox visib - x',
    'Bbox visib - y'
])

axs[1, 1].hist([bbox_all_w, bbox_all_h, bbox_visib_w, bbox_visib_h], bins=20)
axs[1, 1].legend([
    'Bbox all - width',
    'Bbox all - height',
    'Bbox visib - width',
    'Bbox visib - height'
])

f.tight_layout()
plt.show()
