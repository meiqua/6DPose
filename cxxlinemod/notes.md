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