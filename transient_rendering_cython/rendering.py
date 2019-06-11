import sys
sys.path.append('stratified_transient_raytracer')
sys.path.append('embree_intersector')
import numpy as np
import math
import time

import renderer
import embree_intersector

def space_carving_projection(v, space_carving_mesh):
    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (v.shape[0],1))
    barycoord = np.ndarray((v.shape[0],3), dtype=np.float32)

    embree_intersector.embree3_tbb_intersection(v,direction, space_carving_mesh.v,space_carving_mesh.f,barycoord)


    intersection_p = np.ndarray((v.shape[0],3), dtype=np.float32)
    embree_intersector.barycoord_to_world(space_carving_mesh.v,space_carving_mesh.f,barycoord,intersection_p)
    index = barycoord[:,0] >= 0

    v[index,2] = np.maximum(intersection_p[index,2],v[index,2])

 
def inverseRendering(mesh, data, opt):
  measurement_num = opt.lighting.shape[0]

  transient = np.zeros((measurement_num,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')
  gradient = np.zeros((measurement_num, 3*mesh.v.shape[0]), dtype=np.double, order='C')
  
  renderer.renderStreamedGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, opt.w_width, transient, pathlengths, gradient, data)
  return transient, gradient, pathlengths


def forwardRendering(mesh, opt):
  measurement = opt.lighting.shape[0]

  transient = np.zeros((measurement,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')

  if opt.normal == 'fn':
    renderer.renderStreamedTransient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths)
  else:
    renderer.renderStreamedTransientShading(opt.lighting, opt.lighting_normal, mesh.v, mesh.vn, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths)
  return transient, pathlengths    


def face_normal_and_area(v,f):
  p1 = v[f[:,0], :]
  p2 = v[f[:,1], :]
  p3 = v[f[:,2], :]
  n = np.cross(p2 - p1, p3 - p1, axis=1)
  n += sys.float_info.epsilon
  d = np.linalg.norm(n, axis=1)
  n = np.divide(n, np.vstack(d))
  return n, d/2 

def smooth_grad(mesh, smooth_opt):
    s = np.array(mesh.v[:,2])
    s = np.reshape(s, smooth_opt.v_shape)
    Dx_s = 2*s[:,1:-1] - s[:,0:-2] - s[:,2:]
    x = np.hstack((-Dx_s, np.zeros((smooth_opt.v_shape[0],2))))
    x[:,1:-1] += 2*Dx_s
    x[:,2:] -= Dx_s

    Dy_s = 2*s[1:-1:,:] - s[0:-2,:] - s[2:,:]
    y = np.vstack((-Dy_s,np.zeros((2,smooth_opt.v_shape[1]))))
    y[1:-1,:] += 2* Dy_s
    y[2:,:] -= Dy_s

    grad_z = (x + y).flatten()

    grad = np.zeros(mesh.v.shape)
    grad[:,2] += grad_z*smooth_opt.weight

    return grad

def smooth_grad_smooth(mesh, smooth_opt):
    s = np.array(mesh.v[:,2])
    s = np.reshape(s, smooth_opt.v_shape)
    Dx_s = s[:,1:] - s[:,0:-1]
    x = np.hstack((-Dx_s, np.zeros((smooth_opt.v_shape[0],1))))
    x[:,1:] += Dx_s

    Dy_s = s[1:,:] - s[0:-1,:]
    y = np.vstack((-Dy_s,np.zeros((1,smooth_opt.v_shape[1]))))
    y[1:,:] += Dy_s

    grad_z = (x + y).flatten()

    grad = np.zeros(mesh.v.shape)
    grad[:,2] += grad_z*smooth_opt.weight

    return grad

def evaluate_loss(gt_transient, transient, mesh, render_opt, smooth_opt):
  s = np.array(mesh.v[:,2])
  s = np.reshape(s, smooth_opt.v_shape)
  Dx_s = 2*s[:,1:-1] - s[:,0:-2] - s[:,2:]
  Dy_s = 2*s[1:-1,:] - s[0:-2,:] - s[2:,:]
 
  w = np.ones(2*render_opt.w_width+1)/(2*render_opt.w_width+1)
  difference = transient - gt_transient
  for i in range(difference.shape[0]):
    difference[i,:] = np.convolve(difference[i,:], w, 'same')
 
  L1 = np.linalg.norm(difference)**2/difference.shape[0] 
  L2 = (np.linalg.norm(Dx_s)**2 + np.linalg.norm(Dy_s)**2)*smooth_opt.weight/2
  return L1+L2, L1


