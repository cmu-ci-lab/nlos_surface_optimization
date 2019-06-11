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
import torch
from torch import optim
from adam_modified import Adam_Modified
import pyigl as igl
import optimize_parameters

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
    self.albedo_flag = False
    self.alpha_flag = False
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
  testing_flag = 1
  edge_lr_ratio = 1
  loss_flag = 0
  alpha_lr = 0.01 
  gt_mesh = True

  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-.25, .25, 4), np.linspace(-.25, .25, 4))
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
  alpha = 0.1

def optimization(ratio):
	folder_name = os.getcwd() + '/progress-b-2-%f/'% ratio
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/../exp_bunny/setup/bunny_transient.mat'
	setup = scipy.io.loadmat(filename)


	gt_mesh = MESH()
	gt_mesh.v = igl.eigen.MatrixXd(np.array(setup['gt_v'], dtype = np.double))
	gt_mesh.f = igl.eigen.MatrixXd(np.array(setup['gt_f'], dtype = np.double)).castint()

	opt = OPT(20000)

	mesh = MESH()
	mesh_init_location = os.getcwd() + '/init/cnlos_bunny_threshold.obj'

	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

	vertex_num = 400
	grad = rendering.vertex_gradient(mesh, vertex_num, opt)
	grad = grad[:,0]
	delta = 0.001
	transient1, path = rendering.forwardRendering(mesh, opt)
	mesh.v[vertex_num,0] += delta
	transient2, path = rendering.forwardRendering(mesh, opt)
	numerical_grad = (transient2 - transient1)/delta	    



	filename = folder_name + 'data.mat'
	scipy.io.savemat(filename, mdict={'numerical_grad': numerical_grad,'rendered_grad':grad, 'lighting': opt.lighting[0,:], 'vertex_num':vertex_num})
optimization(1)

