import vecadd_ext 

import cupy as cp

shape = (2, 2,3)

a = cp.ones(shape, dtype = cp.float32)
b = cp.ones(shape, dtype = cp.float32)**2
c = cp.zeros(shape, dtype=cp.float32)

vecadd_ext.add_inplace(a,b,c)
print(a,b,c)
assert cp.all(cp.isclose(a+b,c))

import numpy as np

a = np.ones(shape, dtype = np.float32)
b = np.ones(shape, dtype = np.float32)**2
c = np.zeros(shape, dtype=np.float32)

vecadd_ext.add_inplace(a,b,c)
print(a,b,c)

assert np.all(np.isclose(a+b,c))
