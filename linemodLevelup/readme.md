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

Detector: n clusters params  
assign area to feature and hist area info(how many feats in one area) to templ in cropTemplates  
no need for hist, hist when matching  
area pyrDown: clsuter after pyrDown    
similarity_64 & similarity: add to different dst(dst become a vec now) according to featrue area  
active specific area by the thresh of simi.  
class Match: pass active areas to it.  
func match: add a min num of active part param  

!!!!!! num_feat and clusters /4 when pyrDown, may cause problems?  
well, back to /2, /4 or /2, it's a question.  
