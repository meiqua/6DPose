# 6DPose
implement some algorithms of 6d pose estimation  
## pysixd
copied from [sixd_toolkit](https://github.com/thodan/sixd_toolkit)  
deal with model reading/rendering, datasets reading and evaluation  
#### prerequisite
get dataset under 6DPose folder using following cmd  
```
wget -r -np -nH --cut-dirs=1 -R index.html http://ptak.felk.cvut.cz/6DB/public/
```
install opencv3 with contrib rgbd module  
installing by pip may be fine if just use py, 
but compiling from source then installing to conda env is better.  
In my case(ubuntu16.04 with cuda8, 9):  
set OPENCV_EXTRA_MODULES_PATH as /home/meiqua/opencv-build/opencv_contrib/modules  
set CMAKE_INSTALL_PREFIX as /home/meiqua/anaconda3/envs/furnace  
set BUILD_TIFF on (otherwise in c++ there will be a link error, 
caused by tiff version problem)  
set py3 inc/exe/lib to conda env's  
set CUDA_TOOLKITS_ROOT_DIR as /usr/local/cuda-8.0(don't know why, cuda9 can't be compiled with opencv)  

install pybind11
## linemod
Codes in linemod.py will train and detect objects in downloaded dataset.  
Refer to opencv linemod and ork linemod src  
Training and detecting is OK, but extracting and refining pose haven't be done.  
Though detection is fine, but the result is really confusing  
I think the original img should look similar to test scene,
and the predicted center drop in right area, but it seems not that good.
In ork, there is an ICP process following that, maybe ICP helps a lot?  
Well, I'll implement others' paper to compare with linemod without ICP first.  
I think it's caused by scale, all the training set is from R=400mm, and linemod is not
scale invariant  
Amazing(惊了)!when train from rendered img of different R(600 to 1000, per 100), the result is
much better than before. Linemod is quite sensitive to scales.  
However, as is expected, matching time increase dramatically(2.5s per frame now).  

##### Pose refine OK, look pretty good, refer to ork linemod icp.  
![image](./test/results/scene6_match.png)  
  
  
![image2](./test/results/axis.png)

## linemodLevelup
To deal with occlusion and scale problems:  
at each matching pos, scale template depth to scene depth;  
cut template to 4 parts, one of 4's the average response should be above
a threshold. For example, if half of the obj is hidden, original holistic match's
average response will drop to 50%, while part-based match keeps 100%.  

After some effort, 4 parts version works fine, though a lot slower, and bring
more mismatches.  
Maybe just run 4 times is better?  
Well, the scale is a more important problem.  

Attempt2: make low response closer to 0, high response closer to 4.
So the final similarity will depend on high response more. Hope this 
method will distinguish occlusion from mismatch.  
We just need to modify lookup table this time. The result is pretty
interesting: there are 4 levels in ori table. If we change 1,2 to 0,
3 to 1, there are less mismatch in half object case.  
##### image: (half of the obj is occluded)  
![half](./linemodLevelup/test/case1/0000_rgb_half.png)  
##### ori detector: threshold 84 + nms  
![ori](./linemodLevelup/test/case1/result/rgb_half_ori.png)  
##### set low response closer to 0:  
![low to 0](./linemodLevelup/test/case1/result/rgb_half_low_to_0.png)  
##### There are two interesting things:  
modified version have low similarity(which is expected), but it doesn't
matter, we just set threshold lower;  
ori obj bounding circle has an offset caused by the occlusion mask,
while modified one is good.

#### scale experiment
An obvious way to deal with scale problem is scaling template at each
matching position. We tried and show our result below.
Firstly we train templates by rendering img from 600mm, while obj is
about 1000mm in scene. As is expected, our ori detector fails:  
![depth600_ori](./linemodLevelup/test/case1/result/depth600_ori.png)
##### following is ori low to 0 version(simi drops off course):
![depth600_ori_low0](./linemodLevelup/test/case1/result/depth600_ori_low0.png)
![depth600_ori_half](./linemodLevelup/test/case1/result/depth600_ori_half.png)
##### scale template at each match position:
![depth600](./linemodLevelup/test/case1/result/depth600.png)
![depth600_half](./linemodLevelup/test/case1/result/depth600_half.png)

As we can see, scale template does help, but we meet an embarrassed problem:
the match speed drops from 0.03 to 1, nearly 30 times slower... So why
don't we use templates of 30 different scales?  
Well, we may accelerate it by GPU or some magic, but I think there is 
a better way: cut original img to some roughly same depth parts, then 
scale template just one time to each part.  
My insight comes from original speed-up method. So why original detector
is 30 times faster? The magic is in linear memory. because it doesn't scale
template, it can reorganize response map, then we can access response map
in a continuous manner. However, if we need to scale templates at each position, 
the value we want to access looks random now, which is unfriendly to cache.  
However, in fact we don't need to scale templates so frequently, because
linemod is not that sensitive to scales. If we can cut original img to
several almost-same-depths parts, then we just need to scale templates 
several times, and keep continuous access manner in one part.  
  
Amazing! We use a simple histogram to find possible depth, then scale
template at all the depths(typically 5 or so), the result is quite convincing,
 while keep matching time low:  
![depth600_hist](./linemodLevelup/test/case1/result/depth600_hist.png)  
As we can see, the template is trained from 600mm. We use histogram + 1D nms to 
find possible depth in scene, in this case we find 5 possible depths, and 
successfully, 1000mm is one of them. Matching time is about 60ms now.  

Heavily test shows that though detection is fine, but not so accurate as
before, which makes ICP result not so good. I think it's caused by scaling feature 
directly rather than training from img pyramid, it will bring small offset due to pixel 
grid.  

I'll try training from multiple scales, and adopt proper scales when matching.  

Well, if we match from many scales(from 600 to 1800), small scales will disturb us, 
because small obj tends to match random things. So at last I decide to adopt two 
minor modification to linemod:  
modify lookup table, so low response is closer to 0, we expect it can deal with 
occlusion;  
regard different scales(or some prior views?) objs as different objs, 
so when we want to match, we may use some info to determine at which scales we want 
to match.  

Next goal is to improve post-process, like nms, icp ect.



