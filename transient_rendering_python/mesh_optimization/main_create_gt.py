import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time
from scipy.spatial import Delaunay

import rendering

import pyigl as igl

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-2.2, 1.8, 7), np.linspace(-2.2, 1.8, 7))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
 
  measurement_num = lighting.shape[0] 
  #sensor = np.array([0, 0, 0.0])
  #sensor = np.tile(sensor, (measurement_num,1))
  sensor = lighting
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  vn = 0
  doublearea = igl.eigen.MatrixXd()

gt_mesh = MESH()
  
[x, y] = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
x = np.concatenate(x)
y = np.concatenate(y)
z = -0.5 + np.sqrt(4 - np.multiply(x,x) - np.multiply(y,y))
v = np.vstack((x, y, z)).T
vn = -np.vstack((x, y, z+0.5)).T/2
tri = Delaunay(v[:,0:2])


gt_mesh.vn = -np.vstack((x,y,z+0.5)).T/2
gt_mesh.f = igl.eigen.MatrixXi(tri.simplices[:,[0,2,1]])
gt_mesh.v = igl.eigen.MatrixXd(v)

igl.per_face_normals(gt_mesh.v, gt_mesh.f, gt_mesh.fn)
igl.doublearea(gt_mesh.v, gt_mesh.f, gt_mesh.doublearea)

gt_mesh.fn = np.array(gt_mesh.fn)
gt_render_opt = OPT(5000000)
gt_render_opt.normal = 'n'

tic = time.time()
gt_transient = rendering.render_all_collocate(gt_mesh, gt_render_opt)
print(time.time() - tic)

filename = os.getcwd() + '/setup_3.mat'
scipy.io.savemat(filename, mdict={'bin_width': gt_render_opt.distance_resolution, 'lighting': gt_render_opt.lighting, 'sensor':gt_render_opt.sensor, 'gt_v':np.array(gt_mesh.v), 'gt_f': np.array(gt_mesh.f), 'gt_transient': gt_transient})
