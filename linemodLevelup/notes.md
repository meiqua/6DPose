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

## There are two interesting things:  
modified version have low similarity(which is expected), but it doesn't
matter, we just set threshold lower;  
ori obj bounding circle has an offset caused by the occlusion mask,
while modified one is good.

An obvious way to deal with scale problem is scaling template at each
matching position. We tried and show our result below.

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