import numpy as np

def intersect_ray_mesh_batch_directions(origin, direction, mesh, epsilon):
  direction = direction.T

  vertex0 = mesh.v[mesh.f[:,0],:]
  vertex1 = mesh.v[mesh.f[:,1],:]
  vertex2 = mesh.v[mesh.f[:,2],:]
  
  edge1 = vertex0 - vertex2
  edge2 = vertex1 - vertex2

  t = element_multiply(edge2[:,2], direction[1,:]) - element_multiply(edge2[:,1], direction[2,:])
  tvect1 = origin[0] - vertex2[:,0]
  v = element_multiply2(tvect1, t)
  u = element_multiply2(edge1[:,0], t)

  t = element_multiply(edge2[:,0], direction[2,:]) - element_multiply(edge2[:,2], direction[0,:])
  tvect2 = origin[1] - vertex2[:,1]
  v += element_multiply2(tvect2, t)
  u += element_multiply2(edge1[:,1], t)
  
  t = element_multiply(edge2[:,1], direction[0,:]) - element_multiply(edge2[:,0], direction[1,:])
  tvect3 = origin[2] - vertex2[:,2]
  v += element_multiply2(tvect3, t)
  u += element_multiply2(edge1[:,2], t)

  intersect = np.logical_not(abs(u) < epsilon)
  if not intersect.any():
      return

  u[np.logical_not(intersect)] = 1
  t = 1/u
  u = np.multiply(t,v)
  intersect = intersect & (u >= 0) & (u <= 1)    
  if not intersect.any():
      return

  qvect1 = np.multiply(tvect2, edge1[:,2]) - np.multiply(tvect3, edge1[:,1])
  qvect2 = np.multiply(tvect3, edge1[:,0]) - np.multiply(tvect1, edge1[:,2])
  qvect3 = np.multiply(tvect1, edge1[:,1]) - np.multiply(tvect2, edge1[:,0])

  v = element_multiply(qvect1, direction[0,:])
  v += element_multiply(qvect2, direction[1,:])
  v += element_multiply(qvect3, direction[2,:])
  v = np.multiply(t, v)


  tmp = np.multiply(edge2[:,0], qvect1) + np.multiply(edge2[:,1], qvect2) + np.multiply(edge2[:,2], qvect3)
  t = element_multiply2(tmp, t)  
  intersect = intersect & (v >= 0) & (u+v <= 1) 

  return intersect, t, u, v


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
