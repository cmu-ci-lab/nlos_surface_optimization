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

bunny_location = os.getcwd() + '/../../mesh_processing/data/bunny_parallel_T.obj'
gt_mesh = MESH()
igl.readOBJ(bunny_location, gt_mesh.v, gt_mesh.f)

gt_mesh.v = np.array(gt_mesh.v.leftCols(3)/100, dtype=np.float32, order='C')
gt_mesh.v[:,2] = gt_mesh.v[:,2] - 0.5
gt_mesh.f = np.array(gt_mesh.f, dtype=np.int32, order='C')

resolution = 1024
batch = 64


opt = OPT(100000000, resolution)

a = np.arange(opt.lighting.shape[0])
b = np.split(a, batch)
  
directory = os.getcwd() + '/setup/'
if not os.path.isdir(directory):
  os.mkdir(directory)


tmp_opt = OPT(100000000)

for i, x in zip(range(len(b)), b):
  print(i)
  tmp_opt.lighting = opt.lighting[x,:]
  tmp_opt.lighting_normal = opt.lighting_normal[x,:]
    
  tic = time.time()
  gt_transient, pathlength = rendering.forwardRendering(gt_mesh, tmp_opt)
  print('cython transient%f'%(time.time() - tic))


  filename = os.getcwd() + '/setup/bunny_transient_%d_%d.mat'%(resolution,i)
  scipy.io.savemat(filename, mdict={'bin_width': opt.distance_resolution, 'lighting': opt.lighting, 'sensor':opt.sensor, 'gt_v':gt_mesh.v, 'gt_f': gt_mesh.f, 'gt_transient': gt_transient})

