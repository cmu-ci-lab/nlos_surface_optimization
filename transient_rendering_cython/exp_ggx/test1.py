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
  alpha_lr = 0.1 

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
	folder_name = os.getcwd() + '/progress-b-1-%f/'% ratio
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/setup/bunny_transient_64_0.mat'
	setup = scipy.io.loadmat(filename)

	gt_transient = np.array(setup['gt_transient'], dtype=np.double, order = 'C')

	gt_mesh = MESH()
	gt_mesh.v = igl.eigen.MatrixXd(np.array(setup['gt_v'], dtype = np.double))
	gt_mesh.f = igl.eigen.MatrixXd(np.array(setup['gt_f'], dtype = np.double)).castint()

	opt = OPT(20000)
	opt.max_distance_bin = gt_transient.shape[1]
	opt.smooth_weight = 0.001


	mesh = MESH()

	mesh_init_location = os.getcwd() + '/init/cnlos_bunny_threshold_64.obj'
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

	optimization_alpha = torch.Tensor([mesh.alpha])
	optimization_alpha.requires_grad_()
	optimizer_alpha = optim.Adam([optimization_alpha], lr = opt.alpha_lr) 

	dummy_loss2 = optimization_alpha**2
	dummy_loss2.backward()


	global_counter = 0

	for t in range(3):
	  if mesh.f.shape[0] > 250000:
	    break

	  global_counter, l2_record = optimize_parameters.optimize_alpha(mesh, gt_transient, weight, optimization_alpha, optimizer_alpha, opt, 50, global_counter)
	  if t == 0:
	    l2_0 = l2_record

	  lr = (l2_record/l2_0) * lr0 * ((0.99)**(t/50))
	  print('new lr %f' % lr)
	
	  old_v = np.array(mesh.v)
	  global_counter, convergence_flag = optimize_parameters.optimize_shape(mesh, gt_transient, weight, opt, 15, lr, gt_mesh, global_counter)
	  if convergence_flag:
	    if opt.testing_flag == 1:
	        opt.testing_flag = 0
	    else:
	        opt.testing_flag = 1

	  rendering.el_topo_gradient(mesh, old_v)
	  rendering.el_topo_remeshing(mesh, .5/opt.resolution)
	  rendering.isotropic_remeshing(mesh, .5/opt.resolution)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.removeTriangle(mesh,opt)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.border_indicator(mesh)


#TODO
'''

	scale_flag = False
	remesh_flag = False
	weight_flag = True
	alpha_flag = True
	run_count = 0
	for t in range(opt.T):


	        if weight_flag:
	           weight_flag = False
	           print('new smooth weight %f' %opt.smooth_weight)
	           if t > 0:
	    
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'alpha': mesh.alpha, 'transient': transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'smoothing_grad':smoothing_grad, 'sample_num': opt.sample_num})
	    
	    run_count += 1

	    if run_count > 2:
	       if ((l2_original_record[t-1] - original_l2)/l2_original_record[t-1])< opt.loss_epsilon or ((l2_record[t-1] - l2)/l2_record[t-1])< opt.loss_epsilon:
	           if alpha_flag:
	               alpha_flag = False
	               continue
	           if opt.testing_flag == 1:
	              opt.testing_flag = 0
	              opt.smooth_ratio = ratio/10 + t/100 
	              print('shading based')
	           else:
	              alpha_flag = True
	              opt.testing_flag = 1
	              opt.resolution *= 1.5
	              opt.sample_num *= 1.5
	              opt.loss_epsilon /= 2
	              opt.smooth_ratio = ratio + t/10
	              print('remesh %d'%opt.resolution)

	              remesh_flag = True
	           weight_flag = True
	           continue
	    if alpha_flag:
	    else:


	      if run_count == 15:
	          remesh_flag = True
	
	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record, 'l2_original_record': l2_original_record, 'v2_record':v2_record, 'weight':weight})
'''
optimization(100)

