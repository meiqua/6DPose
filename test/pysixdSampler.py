import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pysixd.view_sampler as vSampler
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# pts, pts_level = vSampler.hinter_sampling(60)
# pts = vSampler.fibonacci_sampling(49)

# use views to transform pt 1, 0, 0
views, levels = vSampler.sample_views(50)
pts = np.zeros(shape=(len(views), 3))
for i in range(len(views)):
    view = views[i]
    R = view['R']
    t = view['t']
    testPt = np.matrix([1.0, 0, 0]).T
    temp = R.dot(testPt) + t
    pts[i, :] = temp.reshape(3,)

print(pts.shape)
ax.scatter(xs=pts[:,0], ys=pts[:, 1], zs=pts[:, 2])
plt.show()

print("end of line for break points")
