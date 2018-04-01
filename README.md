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


