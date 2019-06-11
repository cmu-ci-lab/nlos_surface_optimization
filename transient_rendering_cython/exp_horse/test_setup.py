import numpy as np
import scipy.io
import sys, os

sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')
sys.path.append('../exp_bunny')
sys.path.append('../exp_ggx')

import math
import time
import argparse
import rendering
import pyigl as igl
import optimize_parameters

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
    self.gt_mesh = True 
  max_distance_bin = 1200
  distance_resolution = 1.2*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  space_carving_projection = 0
  smooth_weight = 0.0001 
  smooth_ratio = .5
  gamma = 0
  resolution = 64
  loss_epsilon  = 10**-5
  T = 100
  bin_refine_resolution = 10
  sigma_bin = 1
  testing_flag = 0
  edge_lr_ratio = 1
  loss_flag = 0
  alpha_flag = False
  alpha_lr = 0.01 
  albedo_flag = True
  albedo_lr = 1

  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-.35, .35, 64), np.linspace(-.35, .35, 64))
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
  f_affinity = 0
  v_edge = 0
  alpha = 0.5
  albedo = 1

def optimization(ratio):
	lr0 = 0.0001
	lr = lr0
	folder_name = os.getcwd() + '/test/'
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/setup/horse_dataset_calibrated.mat'
	setup = scipy.io.loadmat(filename)

	gt_transient = np.array(setup['transientsCalibrated'].T, dtype=np.double, order = 'C')

	opt = OPT(20000)
	opt.max_distance_bin = gt_transient.shape[1]
	opt.smooth_weight = 0.001
	opt.lighting = np.array(setup['coords'].T, dtype=np.float32, order='C')
	opt.sensor = np.array(setup['coords'].T, dtype=np.float32, order='C')
	opt.gt_mesh = False

	gt_mesh = MESH()
	mesh = MESH()
	mesh_init_location = os.getcwd() + '/setup/cnlos_horse_threshold.obj'
	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

	transient, pathlength = rendering.forwardRendering(mesh, opt)		
	

	filename = folder_name + '1.mat'
	scipy.io.savemat(filename, mdict={'transient':transient})

optimization(50)

