import numpy as np
import scipy.io
import sys, os

sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')
sys.path.append('../exp_bunny')

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
  alpha_flag = True
  alpha_lr = 0.01 

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
  f_affinity = 0
  v_edge = 0
  alpha = 0.5

def optimization(ratio):
	lr0 = 0.0001
	lr = lr0
	folder_name = os.getcwd() + '/progress-b-3-%f/'% ratio
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	test_num = 1
	filename = os.getcwd() + '/setup/bunny_transient_64_0_%d.mat'%test_num
	setup = scipy.io.loadmat(filename)

	gt_transient = np.array(setup['gt_transient'], dtype=np.double, order = 'C')

	gt_mesh = MESH()
	gt_mesh.v = igl.eigen.MatrixXd(np.array(setup['gt_v'], dtype = np.double))
	gt_mesh.f = igl.eigen.MatrixXd(np.array(setup['gt_f'], dtype = np.double)).castint()

	opt = OPT(20000)
	opt.max_distance_bin = gt_transient.shape[1]
	opt.smooth_weight = 0.001


	mesh = MESH()

	mesh_init_location = os.getcwd() + '/init/cnlos_bunny_threshold_64_0_%d.obj'%test_num
	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

	opt.resolution = 64
	opt.smooth_ratio = ratio
      
	rendering.isotropic_remeshing(mesh, .5/opt.resolution)
	old_v = np.array(mesh.v)

	rendering.compute_mesh_affinity(mesh)
	rendering.border_indicator(mesh)


	weight = rendering.create_weighting_function(gt_transient, opt.gamma)



	global_counter = 0


	l2 = np.empty(500)
	alpha = np.empty(500)
	for t in range(500):
	  if mesh.f.shape[0] > 250000:
	    break

	  global_counter, l2_record = optimize_parameters.optimize_alpha(mesh, gt_transient, weight, opt, 50, global_counter)
	  if t == 0:
	    l2_0 = l2_record

	  lr = (l2_record/l2_0) * lr0 * ((0.99)**(t/2))
	  print('new lr %f' % lr)
	
	  old_v = np.array(mesh.v)
	  global_counter, convergence_flag, l2_record = optimize_parameters.optimize_shape(mesh, gt_transient, weight, opt, 15, lr, gt_mesh, global_counter)
	  if convergence_flag:
	    if opt.testing_flag == 1:
	        opt.testing_flag = 0
	        opt.smooth_ratio = ratio/10 + t/100 
	        print('shading')
	    else:
	        opt.testing_flag = 1
	        opt.smooth_ratio = ratio + t/10

	  rendering.el_topo_gradient(mesh, old_v)
	  rendering.el_topo_remeshing(mesh, .5/opt.resolution)
	  rendering.isotropic_remeshing(mesh, .5/opt.resolution)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.removeTriangle(mesh,opt)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.border_indicator(mesh)
	    
	  filename = folder_name + '%05d.mat'%(t)
	  scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'alpha': mesh.alpha})
	  l2[t] = l2_record
	  alpha[t] = mesh.alpha

	filename = folder_name + 'progress.mat'
	scipy.io.savemat(filename, mdict={'alpha':alpha, 'l2': l2})
	  #if t >= 1:
	  #  if (l2[t-1] - l2[t]/l2[t-1]) < opt.epsilon:
	  #    break


#TODO
'''

	    
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'alpha': mesh.alpha, 'transient': transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'smoothing_grad':smoothing_grad, 'sample_num': opt.sample_num})
	    
	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record, 'l2_original_record': l2_original_record, 'v2_record':v2_record, 'weight':weight})
'''
optimization(50)

