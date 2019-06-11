import numpy as np
def element_divide2(a,b):
  sz = a.shape
  if b.size == sz[0]:
      b = np.tile(b, (sz[1],1)).T
  else:
      b = np.tile(b, (sz[0],1))
  return np.divide(a,b)

def element_multiply2(a,b):
  sz = b.shape
  if a.size == sz[0]:
    a = np.tile(a, (sz[1],1)).T
  else:
    a = np.tile(a, (sz[0],1))
  return np.multiply(a,b)


def element_multiply(a,b):
  a = np.tile(a, (b.size, 1)).T
  b = np.tile(b, (a.shape[0], 1))
  return np.multiply(a,b)
