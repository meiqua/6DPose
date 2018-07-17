Though patch version linemod should be easy to implement, we found some problems
when testing:  
1. select scatterd feature func will select features too close if there are few
feature, so we constrain min distance of two features to 2 pixels, otherwise
return false
2. spread ori can only work with 16*n cols and rows, so we just pad zero to  16*n for
our patch. A very strange thing is that for single channel img, step1()!=cols in
some case, this will break spread func. We change step1() to cols then it works fine
3. for normals extraction, if we have no zero padding(found via trial and error T_T), which is our case when using
patches, all normals will become 0. We pad it first then crop at last.  
strange proto bug, make twice then everything is OK  
