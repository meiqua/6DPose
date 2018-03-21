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
