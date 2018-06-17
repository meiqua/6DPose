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