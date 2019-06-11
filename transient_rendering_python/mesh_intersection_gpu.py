import numpy as np
import torch
def intersect_ray_mesh_batch_directions(origin, direction, mesh, epsilon):
  direction = direction.transpose(0,1)

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

  intersect = (torch.abs(u) < epsilon)^1
  if not intersect.any():
      return intersect, t, u, v

  u[intersect^1] = 1
  t = 1/u
  u = torch.mul(t,v)
  intersect = intersect.__and__(u >= 0).__and__(u <= 1)    
  if not intersect.any():
      return intersect, t, u, v


  qvect1 = torch.mul(tvect2, edge1[:,2]) - torch.mul(tvect3, edge1[:,1])
  qvect2 = torch.mul(tvect3, edge1[:,0]) - torch.mul(tvect1, edge1[:,2])
  qvect3 = torch.mul(tvect1, edge1[:,1]) - torch.mul(tvect2, edge1[:,0])

  v = element_multiply(qvect1, direction[0,:])
  v += element_multiply(qvect2, direction[1,:])
  v += element_multiply(qvect3, direction[2,:])
  
  v = torch.mul(t, v)


  tmp = torch.mul(edge2[:,0], qvect1) + torch.mul(edge2[:,1], qvect2) + torch.mul(edge2[:,2], qvect3)
  t = element_multiply2(tmp, t)  
  intersect = intersect.__and__(v >= 0).__and__ (u+v <= 1) 
  
  return intersect, t, u, v


def element_divide2(a,b):

  b = torch.transpose(b.repeat(a.size(1),1),0,1)
  return torch.div(a,b)

def element_multiply2(a,b):
  if a.size(0) == b.size(0):
    a = torch.transpose(a.repeat(b.size(1),1),0,1)
  else:
    a = a.repeat(b.size(0),1)
  return torch.mul(a,b)


def element_multiply(a,b):
  a = torch.transpose(a.repeat(b.size(0),1),0,1)
  b = b.repeat(a.size(0),1)
  return torch.mul(a,b)
   
