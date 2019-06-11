import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')

import math
import time
import scipy
from scipy.spatial import Delaunay

import rendering

import pyigl as igl

class OPT:
  def __init__(self, sample_num=2500, resolution=64):
    self.sample_num = sample_num
    self.resolution = resolution
    [lighting_x, lighting_y] = np.meshgrid(np.linspace(-.25, .25, self.resolution), np.linspace(-.25, .25, self.resolution))
    lighting_x = np.concatenate(lighting_x)
    lighting_y = np.concatenate(lighting_y)
    lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
    self.lighting = np.array(lighting, dtype = np.float32, order='C')

    measurement_num = lighting.shape[0] 
    self.sensor = lighting
    sensor_normal = np.array([0, 0, 1], dtype = np.float32, order='C')
    self.sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
    lighting_normal = np.array([0, 0, 1], dtype = np.float32, order='C')
    self.lighting_normal = np.tile(lighting_normal, (measurement_num,1))
  max_distance_bin = 1200
  distance_resolution = 1.2*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = 0
  vn = 0
  face_area = 0

bunny_location = os.getcwd() + '/setup/armadillo.obj'
gt_mesh = MESH()
igl.readOBJ(bunny_location, gt_mesh.v, gt_mesh.f)

gt_mesh.v = np.array(gt_mesh.v, dtype=np.float32, order='C')
gt_mesh.f = np.array(gt_mesh.f, dtype=np.int32, order='C')

#gt_render_opt = OPT(10000, 256)
gt_render_opt = OPT(100000000, 512)

tic = time.time()
gt_transient, pathlength = rendering.forwardRendering(gt_mesh, gt_render_opt, 16)
print('cython transient%f'%(time.time() - tic))

directory = os.getcwd() + '/setup/'
if not os.path.isdir(directory):
  os.mkdir(directory)

filename = os.getcwd() + '/setup/armadillo_transient.mat'
scipy.io.savemat(filename, mdict={'bin_width': gt_render_opt.distance_resolution, 'lighting': gt_render_opt.lighting, 'sensor':gt_render_opt.sensor, 'gt_v':gt_mesh.v, 'gt_f': gt_mesh.f, 'gt_transient': gt_transient})
