import vecadd_ext 

import cupy as cp

a = cp.arange(3, dtype = cp.float32)
b = cp.arange(3, dtype = cp.float32)**2
c = cp.zeros(3, dtype=cp.float32)

vecadd_ext.add_inplace(a,b,c)
print(a,b,c)
assert cp.all(cp.isclose(a+b,c))

import numpy as np

a = np.arange(3, dtype = np.float32)
b = np.arange(3, dtype = np.float32)**2
c = np.zeros(3, dtype=np.float32)

vecadd_ext.add_inplace(a,b,c)
print(a,b,c)

assert np.all(np.isclose(a+b,c))
