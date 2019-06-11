import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time
import argparse
import rendering
import torch
from torch import optim

import pyigl as igl

class SMOOTH_OPT:
  v_shape = (64,64)
  #weight = 0.00001 
  weight = 0 

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1024
  distance_resolution = 1.2*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  space_carving_projection = 1
  w_width = 0
  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-.25, .25, 64), np.linspace(-.25, .25, 64))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
  lighting = np.array(lighting, dtype = np.float32, order='C')
 
  measurement_num = lighting.shape[0] 
  sensor = lighting
  sensor_normal = np.array([0, 0, 1], dtype = np.float32, order='C')
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1], dtype = np.float32, order='C')
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  vn = 0
folder_name = os.getcwd() + '/gradient_test/'
if not os.path.isdir(folder_name):
  os.mkdir(folder_name)

filename = os.getcwd() + '/setup/bunny_transient.mat'
setup = scipy.io.loadmat(filename)

gt_transient = np.array(setup['gt_transient'], dtype=np.double, order = 'C')

mesh = MESH()
mesh_init_location = os.getcwd() + '/cnlos_bunny_threshold.obj'
igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

T = 10
#T = 150
for power in range(4,7):

	opt = OPT(10**power)

	for t in range(T):
	    print('t = %d'% t)    
	    tic = time.time()
	    transient, grad, pathlength = rendering.inverseRendering(mesh, gt_transient, opt)	    
	    print(time.time() - tic)
	    filename = folder_name + 'new_%05d_%05d.mat'%(power, t)
	    scipy.io.savemat(filename, mdict={'grad': grad, 'sample_num': opt.sample_num})

