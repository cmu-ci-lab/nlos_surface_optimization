import sys
#sys.path.append('../stratified_transient_raytracer')
sys.path.append('../smoothed_transient')
sys.path.append('../embree_intersector')
sys.path.append('../cgal_api')
sys.path.append('../el_topo_api')
sys.path.append('../ggx')
sys.path.append('../jitter')

sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')
#sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import numpy as np
import math
import time
from scipy.spatial import Delaunay

import ggx
import renderer
import embree_intersector
import cgal_api
import el_topo_api
import jitter

import pyigl as igl

def vertex_gradient(mesh, vertex_num, opt):
    gradient = np.zeros((opt.max_distance_bin,3), dtype=np.double, order = 'C')

    renderer.renderStreamedVertexGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, gradient, vertex_num, opt.bin_refine_resolution, opt.sigma_bin)
    return gradient

def el_topo_remeshing(mesh, target_edge_length):
  v_prep_num = 10*mesh.v.shape[0]
  f_prep_num = 10*mesh.f.shape[0]
  v = np.empty((v_prep_num,3), dtype=np.double, order='C')
  f = np.empty((f_prep_num,3), dtype = np.int32, order='C')
  
  new_v_num, new_f_num = el_topo_api.el_topo_remesh(mesh.v, mesh.f, v, f, 0, 0, target_edge_length)
  if new_v_num > v_prep_num:
    print('v not enough')
    return False
  if new_f_num > f_prep_num:
    print('f not enough')
    return False

  mesh.v = np.array(v[range(new_v_num),:], dtype=np.float32, order='C')
  mesh.f = f[range(new_f_num),:]
  return True


def el_topo_gradient(mesh, old_v):
  new_v = np.array(mesh.v, dtype=np.double, order='C')
  v = np.array(old_v, dtype=np.double, order='C')
  
  el_topo_api.el_topo_gradient(v, mesh.f, new_v)
  mesh.v = new_v

def keep_largest_connected_component(mesh):
  v_num, f_num = cgal_api.keep_largest_connected_component(mesh.v, mesh.f, 0, 0)
  mesh.v = mesh.v[range(v_num),:]
  mesh.f = mesh.f[range(f_num),:]

def compute_mesh_affinity(mesh):
  mesh.f_affinity = -1* np.ones(mesh.f.shape, dtype=np.int32, order='C')
  cgal_api.face_affinity(mesh.v, mesh.f, mesh.f_affinity)

def border_indicator(mesh):
  v_idx = np.zeros(mesh.v.shape[0], dtype=np.int32, order='C')
  cgal_api.border_vertex(mesh.v, mesh.f, v_idx)
  mesh.v_edge = v_idx

def isotropic_remeshing(mesh, target_edge_length, flag=0):
  if flag == 0 :
    v_prep_num = 5*mesh.v.shape[0]
    f_prep_num = 5*mesh.f.shape[0]
  else:
    v_prep_num = 400*mesh.v.shape[0]
    f_prep_num = 400*mesh.f.shape[0]

  v = np.empty((v_prep_num,3), dtype=np.float32, order='C')
  f = np.empty((f_prep_num,3), dtype = np.int32, order='C')
  nb_iter = 3
  new_v_num, new_f_num = cgal_api.isotropic_remeshing(mesh.v, mesh.f, target_edge_length, nb_iter, 0, 0, v, f)
  if new_v_num > v_prep_num:
    print('v not enough')
    return False
  if new_f_num > f_prep_num:
    print('f not enough')
    return False

  mesh.v = v[range(new_v_num),:]
  mesh.f = f[range(new_f_num),:]
  return True

def upsample(mesh):
  v = igl.eigen.MatrixXd(mesh.v.astype(np.double))
  f = igl.eigen.MatrixXi(mesh.f)
  igl.upsample(v,f)
  mesh.v = np.array(v, dtype=np.float32, order='C')
  mesh.f = np.array(f, dtype=np.int32, order='C')
 

def recompute_connectivity(mesh): 
    tri = Delaunay(mesh.v[:,0:2])
    new_f = np.array(tri.simplices[:,[0,2,1]], dtype=np.int32, order='C')
    valid = np.zeros(new_f.shape[0])

    v1 = mesh.v[new_f[:,0],:]
    v2 = mesh.v[new_f[:,1],:]
    v3 = mesh.v[new_f[:,2],:]
    v = (v1+v2+v3)/3
    v[:,2] = 0  
    v = np.array(v, dtype=np.float32, order='C')

    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (v.shape[0],1))

    barycoord = np.ndarray(v.shape[0], dtype=np.float32, order='C')
    embree_intersector.embree3_tbb_short_intersection(v, direction, mesh.v, mesh.f, barycoord)
    '''
    v = mesh.v[:,0:2]
    v = np.hstack((v,np.zeros((v.shape[0],1))))
    v = np.array(v, dtype=np.float32,order='C')
    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (v.shape[0],1))

    barycoord2 = np.ndarray(v.shape, dtype=np.float32, order='C')
    embree_intersector.embree3_tbb_intersection(v, direction, mesh.v, mesh.f, barycoord2)
    
    intersection_p = np.ndarray(v.shape, dtype=np.float32, order='C')
    embree_intersector.barycoord_to_world(mesh.v, mesh.f, barycoord2, intersection_p)
    
    mask = barycoord2[:,0] >= 0
    mesh.v[mask,:] = intersection_p[mask,:] 
    '''
    mesh.f = new_f[barycoord >=0,:]

def remesh(mesh, res):
    lower_bound = [-.25, -.25]
    upper_bound = [.25, .25]
    [o_x, o_y] = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], res), np.linspace(lower_bound[1], upper_bound[1],res))
    o_x = np.concatenate(o_x)
    o_y = np.concatenate(o_y)
    o = np.vstack((o_x,o_y,np.zeros_like(o_x))).T
    o = np.array(o, dtype=np.float32, order = 'C')
    
    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (o.shape[0],1))
    barycoord = np.ndarray(o.shape, dtype=np.float32, order='C')

    embree_intersector.embree3_tbb_intersection(o, direction, mesh.v, mesh.f, barycoord)

    intersection_p = np.ndarray(o.shape, dtype=np.float32, order='C')
    embree_intersector.barycoord_to_world(mesh.v, mesh.f, barycoord, intersection_p)
    
    mask = barycoord[:,0] >= 0
    intersection_p = intersection_p[mask,:]
    edge_v = mesh.v[mesh.v_edge==1,:]



    new_v = np.vstack((intersection_p, edge_v))
    tri = Delaunay(new_v[:,0:2])
    new_f = np.array(tri.simplices[:,[0,2,1]], dtype=np.int32, order='C')
    
    v1 = new_v[new_f[:,0],:]
    v2 = new_v[new_f[:,1],:]
    v3 = new_v[new_f[:,2],:]
    v = (v1+v2+v3)/3
    v[:,2] = 0  
    v = np.array(v, dtype=np.float32, order='C')

    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (v.shape[0],1))

    barycoord = np.ndarray(v.shape[0], dtype=np.float32, order='C')
    embree_intersector.embree3_tbb_short_intersection(v, direction, mesh.v, mesh.f, barycoord)
    mesh.v = new_v
    mesh.f = new_f[barycoord >= 0,:]


def compute_v2(v, gt_mesh):
    new_v = igl.eigen.MatrixXd(np.array(v,dtype=np.double))
    S = igl.eigen.MatrixXd()
    I = igl.eigen.MatrixXi()
    C = igl.eigen.MatrixXd()
    N = igl.eigen.MatrixXd()

    igl.signed_distance(new_v, gt_mesh.v, gt_mesh.f, igl.SignedDistanceType.SIGNED_DISTANCE_TYPE_UNSIGNED, S, I, C, N)
    distance = np.array(S)
    return np.average(distance) 

def space_carving_projection(v, space_carving_mesh):
    direction = np.array([0,0,1], dtype=np.float32, order='C')
    direction = np.tile(direction, (v.shape[0],1))
    barycoord = np.ndarray((v.shape[0],3), dtype=np.float32, order='C')

    new_v = np.array(v)
    new_v[:,2] = 0
    embree_intersector.embree3_tbb_intersection(new_v,direction, space_carving_mesh.v,space_carving_mesh.f,barycoord)

    intersection_p = np.ndarray((v.shape[0],3), dtype=np.float32, order='C')
    embree_intersector.barycoord_to_world(space_carving_mesh.v,space_carving_mesh.f,barycoord,intersection_p)
    index = barycoord[:,0] >= 0

    v[index,2] = np.maximum(intersection_p[index,2],v[index,2])

def create_weighting_function(data, gamma=1):
  eps = 0.1
  #i_max = np.max(data,axis=1)
  i_max = np.max(data)
  normalized_data = data/i_max
  weight = (normalized_data + eps)**gamma
  total = np.sum(weight)
  weight = weight/total
  weight *= data.shape[0]*data.shape[1]
  return weight

def inverseShadingRendering(mesh, data, weight, opt):
  mesh.vn = np.empty(mesh.v.shape, dtype=np.float32, order='C')
  cgal_api.per_vertex_normal(mesh.v, mesh.f, mesh.vn)

  measurement_num = opt.lighting.shape[0]

  transient = np.zeros((measurement_num,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')
  gradient = np.zeros(mesh.v.shape, dtype=np.double, order='C')
  renderer.renderStreamedShadingGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, mesh.vn, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, gradient, data, weight, opt.bin_refine_resolution, opt.sigma_bin, opt.testing_flag)
  return transient, gradient, pathlengths


def inverseRenderingAlpha(mesh, data, weight, opt):
  measurement_num = opt.lighting.shape[0]
  transient = np.zeros((measurement_num,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')

  g = ggx.renderStreamedGradientAlpha(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, mesh.alpha, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, data, weight, opt.bin_refine_resolution, opt.sigma_bin)
  return transient, g


def inverseRenderingAlbedo(mesh, data, weight, opt):
  measurement_num = opt.lighting.shape[0]

  transient = np.zeros((measurement_num,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')
  albedo = np.ones(mesh.v.shape[0], dtype=np.float32, order='C') * mesh.albedo

  g = renderer.renderStreamedGradientAlbedo(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, albedo, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, data, weight, opt.bin_refine_resolution, opt.sigma_bin, opt.testing_flag, opt.loss_flag)

  return transient, g
 
def inverseRendering(mesh, data, weight, opt):
  measurement_num = opt.lighting.shape[0]

  transient = np.zeros((measurement_num,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')
  gradient = np.zeros(mesh.v.shape, dtype=np.double, order='C')

  if opt.alpha_flag:
    ggx.renderStreamedGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, mesh.alpha, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, gradient, data, weight, opt.bin_refine_resolution, opt.sigma_bin, opt.testing_flag)
  else:
    if opt.jitter:
        jitter.renderStreamedGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin*opt.distance_resolution, opt.distance_resolution, opt.jitter_weight, opt.jitter_grad, opt.jitter_offset, transient, pathlengths, gradient, data, weight, opt.testing_flag) 
    elif opt.albedo_flag:
      albedo = np.ones(mesh.v.shape[0], dtype=np.float32, order='C') * mesh.albedo
      renderer.renderStreamedGradientWithAlbedo(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, albedo, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, gradient, data, weight, opt.bin_refine_resolution, opt.sigma_bin, opt.testing_flag, opt.loss_flag)
    else:
      renderer.renderStreamedGradient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, gradient, data, weight, opt.bin_refine_resolution, opt.sigma_bin, opt.testing_flag, opt.loss_flag)
  return transient, gradient, pathlengths

def removeTriangle(mesh, opt):
  intensity = np.zeros(mesh.f.shape[0], dtype=np.double, order='C')
  renderer.renderStreamedTriangleIntensity(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, intensity)

  threshold = 0
  keep_face = np.logical_or((intensity > threshold), np.sum(mesh.f_affinity < 0, axis=1) == 0)
  print('remove #face:%d'%(mesh.f.shape[0]-np.sum(keep_face)))
  mesh.f = mesh.f[keep_face,:]

def forwardRendering(mesh, opt):
  measurement = opt.lighting.shape[0]

  transient = np.zeros((measurement,opt.max_distance_bin), dtype=np.double, order = 'C')
  pathlengths = np.zeros(opt.max_distance_bin, dtype=np.double, order = 'C')

  if opt.alpha_flag:
    if opt.normal == 'fn':
      ggx.renderStreamedTransient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, mesh.alpha, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, 1, 1)
    else:
      ggx.renderStreamedTransientShading(opt.lighting, opt.lighting_normal, mesh.v, mesh.vn, mesh.f, mesh.alpha, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths)
  else:
    if opt.normal == 'fn':
      renderer.renderStreamedTransient(opt.lighting, opt.lighting_normal, mesh.v, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths, 1, 1)
    else:
      renderer.renderStreamedTransientShading(opt.lighting, opt.lighting_normal, mesh.v, mesh.vn, mesh.f, opt.sample_num, 0, opt.max_distance_bin * opt.distance_resolution, opt.distance_resolution, transient, pathlengths)

  return transient, pathlengths    


def renderStreamedNormalSmoothing(mesh): 
  gradient = np.zeros(mesh.v.shape, dtype=np.double, order='C')
  val = renderer.renderStreamedNormalSmoothing(mesh.v, mesh.f, mesh.f_affinity, gradient)
  return val, gradient

def renderStreamedCurvatureGradient(mesh): 
  gradient = np.zeros(mesh.v.shape, dtype=np.double, order='C')
  renderer.renderStreamedCurvatureGradient(mesh.v, mesh.f, gradient)
  return gradient

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



def evaluate_loss_with_normal_smoothness(gt_transient, weight, transient, smoothing_val, mesh, render_opt):
  difference = transient - gt_transient
  difference *= np.sqrt(weight) 
  L1 = np.linalg.norm(difference)**2/difference.shape[0] 

  L2 = render_opt.smooth_weight*smoothing_val

  return L1+L2, L1

def evaluate_loss_with_curvature(gt_transient, weight, transient, mesh, render_opt):
  difference = transient - gt_transient
  #L3 = np.linalg.norm(difference)**2/difference.shape[0] 
  difference *= np.sqrt(weight) 
  L1 = np.linalg.norm(difference)**2/difference.shape[0] 

  n, area = face_normal_and_area(mesh.v, mesh.f)
  total_area = sum(area)
  L2 = render_opt.smooth_weight*total_area

  return L1+L2, L1, total_area


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


