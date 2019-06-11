import numpy as np
import torch
from itertools import compress
from torch.autograd import Variable
import pyigl as igl

import element_wise_manipulation

def find_triangle(mesh, direction, lighting, sensor):
    sample_num = direction.shape[0]
    source = igl.eigen.MatrixXd(np.tile(lighting, (sample_num,1)))
    n = igl.eigen.MatrixXd(direction)

    barycoord = igl.eigen.MatrixXd()
    barycoord = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)
    fid = np.array(barycoord.col(0).transpose())[0]
    idx = list(compress(range(sample_num), fid != -1))

    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)
    intersection_p = np.array(intersection_p)

    v2 = np.empty((sample_num, 3))
    d2 = np.empty(sample_num)

    v2[idx,:] = sensor - intersection_p[idx,:]
    d2[idx] = np.sqrt(np.sum(v2[idx,:]**2, axis = 1)) 
    v2[idx,:] = element_wise_manipulation.element_divide2(v2[idx,:], d2[idx])

    source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx), 1)))
    n = igl.eigen.MatrixXd(-v2[idx,:])

    barycoord2 = igl.eigen.MatrixXd()
    barycoord2 = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)
    fid2 = np.array(barycoord2.col(0).transpose())[0]

    idx = list(compress(idx, fid[idx] != fid2))
    fid[idx] = -1

    return fid


def intersect_ray_mesh_one_direction(mesh, direction, triangle, origin):
  vertex0 = torch.squeeze(mesh.v[triangle[0],:])
  vertex1 = torch.squeeze(mesh.v[triangle[1],:])
  vertex2 = torch.squeeze(mesh.v[triangle[2],:])

  edge1 = vertex0 - vertex2
  edge2 = vertex1 - vertex2

  t = edge2[2]*direction[1] - edge2[1]*direction[2]
  
  tvect1 = origin[0] - vertex2[0]
  v = t* tvect1 
  u = edge1[0]*t
  t = edge2[0] * direction[2] - edge2[2] * direction[0]
  tvect2 = origin[1] - vertex2[1]
  v += tvect2*t
  u += edge1[1]*t
  t = edge2[1]*direction[0] - edge2[0]*direction[1]
  tvect3 = origin[2] - vertex2[2]
  v += tvect3*t
  u += edge1[2]*t

  t = 1/u
  u = t*v
  
  qvect1 = tvect2*edge1[2] - tvect3 * edge1[1]
  qvect2 = tvect3*edge1[0] - tvect1 * edge1[2]
  qvect3 = tvect1*edge1[1] - tvect2 * edge1[0]

  
  v = qvect1*direction[0] + qvect2*direction[1] + qvect3*direction[2]

  v = torch.mul(t,v)
  
  tmp = edge2[0]*qvect1 + edge2[1]*qvect2 + edge2[2]*qvect3
  t = torch.mul(t,tmp)
  return u, v, t

  

