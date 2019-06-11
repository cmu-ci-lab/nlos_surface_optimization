import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time

import rotation_matrix
import rendering
import torch
from torch import optim

import pyigl as igl

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
 
  measurement_num = lighting.shape[0] 
  sensor = np.array([0, 0, 0.0])
  sensor = np.tile(sensor, (measurement_num,1))
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

#mesh_location = '../mesh_processing/data/bunny.obj'
gt_mesh = MESH()
gt_mesh.v = igl.eigen.MatrixXd([[-1,-1,0.9],[1, -1, 1],[1, 1, 1.2],[-1, 1, 1],[-2, -2, 1], [2,1, 1]])
gt_mesh.f = igl.eigen.MatrixXd([[0,2,1],[0, 3,2], [4,3,0], [1,2,5]]).castint()
igl.per_face_normals(gt_mesh.v, gt_mesh.f, gt_mesh.fn)

gt_render_opt = OPT(10000000)

tic = time.time()
gt_transient = rendering.render_all(gt_mesh, gt_render_opt)
print(time.time() - tic)


mesh = MESH()
mesh.v = np.array(gt_mesh.v)
mesh.v += np.random.normal(0, 0.2, mesh.v.shape)


filename = os.getcwd() + '/setup.mat'
scipy.io.savemat(filename, mdict={'gt_v':np.array(gt_mesh.v), 'gt_f': np.array(gt_mesh.f), 'v':np.array(mesh.v),  'gt_transient': gt_transient})
