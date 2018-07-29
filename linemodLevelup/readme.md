### image: (half of the obj is occluded)  

![half](./test/case1/0000_rgb_half.png)  

### ori detector: threshold 84 + nms  

![ori](./test/case1/result/rgb_half_ori.png)  

### set low response closer to 0  

![low to 0](./test/case1/result/rgb_half_low_to_0.png)  

### scale experiment

Firstly we train templates by rendering img from 600mm, while obj is
about 1000mm in scene. As is expected, our ori detector fails:  
![depth600_ori](./test/case1/result/depth600_ori.png)  

### ori low to 0 version(simi drops off course)

![depth600_ori_low0](./test/case1/result/depth600_ori_low0.png)
![depth600_ori_half](./test/case1/result/depth600_ori_half.png)

### scale template at each match position, good but 10 times slower

![depth600](./test/case1/result/depth600.png)
![depth600_half](./test/case1/result/depth600_half.png)

### histogram + 1D nms to find primary depth  

![depth600_hist](./test/case1/result/depth600_hist.png)  
As we can see, the template is trained from 600mm. We use histogram + 1D nms to
find possible depth in scene, in this case we find 5 possible depths, and
successfully, 1000mm is one of them. Matching time is about 60ms now.  

todo: more than 64 features. need to modify
similarity(local), and addSimilarities(delete 8u 8u), and distinguish them
because use 16 sse may be slower than 8 sse  
DONE  

todo:  
spread step same as stride or 2:1?  
PSO or some other method rather than ICP  

may implement PSO:  
Correspondence-free pose estimation for 3D objects from noisy depth data  

1D nms to select primary depth, or cut to some primary depth boxes?  

nms for accurate edge  

with selective search? selective search in depth modality?  

2bit label _mm_max_epi8

