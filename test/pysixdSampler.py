import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pysixd.view_sampler as vSampler

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# pts, pts_level = vSampler.hinter_sampling(60)
pts = vSampler.fibonacci_sampling(49)
print(pts.shape)
ax.scatter(xs=pts[:,0], ys=pts[:,1], zs=pts[:,2])
plt.show()

print("end of line for break points")