import numpy as np
import cxxlinemod as eb
import copy
a = eb.read_image("test.png")
print('init a: 0x%x' % id(a))
eb.show_image(a)  # work

# Proves that it's still the same thing
b = eb.passthru(a)
print('same b: 0x%x' % id(b))

# Make a copy
c = eb.clone(b)
print('diff c: 0x%x' % id(c))

d=copy.deepcopy(c)
eb.show_image(d)  # still works
print('diff d: 0x%x' % id(d))

# different allocator
e = np.zeros(shape=(100,100), dtype=np.uint8)
print('\ninit e: 0x%x' % id(e))

f = eb.passthru(e)
print('same f: 0x%x' % id(f))

