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
  albedo_lr0 = .05
  albedo_lr = .05
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
	lr0 = 0.00005
	lr = lr0
	folder_name = os.getcwd() + '/progress-b-23-%f/'% ratio
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

	opt.resolution = 64
	opt.smooth_ratio = ratio
      
	rendering.isotropic_remeshing(mesh, .5/opt.resolution)
	old_v = np.array(mesh.v)

	rendering.compute_mesh_affinity(mesh)
	rendering.border_indicator(mesh)

	weight = rendering.create_weighting_function(gt_transient, opt.gamma)

	#mesh.albedo = optimize_parameters.initial_fitting_albedo(mesh, gt_transient, weight, opt)

	global_counter = 0
	l2 = np.empty(400)
	albedo = np.empty(400)
	resolution_cnt = 0
	for t in range(400):
	  if mesh.f.shape[0] > 250000:
	    break

	  #global_counter, l2_record = optimize_parameters.optimize_albedo(mesh, gt_transient, weight, opt, 50, global_counter, folder_name)

	
	  old_v = np.array(mesh.v)
	  global_counter, convergence_flag, l2_record = optimize_parameters.optimize_shape(mesh, gt_transient, weight, opt, 15, lr, gt_mesh, global_counter, folder_name)
	  #opt.albedo_lr = (l2_record/l2_0) * opt.albedo_lr0 *  ((0.99)**(global_counter/20))
	  if t == 0:
	    l2_0 = l2_record

	  lr = (l2_record/l2_0) * lr0 * ((0.99)**(global_counter/20))
	  print('new lr %f' % lr)


	  if convergence_flag:
	    if opt.testing_flag == 1:
	        opt.testing_flag = 0
	        opt.smooth_ratio = ratio/10 + global_counter/100 
	        print('shading')
	    else:
	        opt.testing_flag = 1
	        opt.smooth_ratio = ratio + global_counter/10
	        opt.resolution *= 1.5
	        opt.sample_num *= 1.5

	  rendering.el_topo_gradient(mesh, old_v)
	  rendering.el_topo_remeshing(mesh, .8/opt.resolution)
	  rendering.isotropic_remeshing(mesh, .8/opt.resolution)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.removeTriangle(mesh,opt)

	  rendering.compute_mesh_affinity(mesh)
	  rendering.border_indicator(mesh)
	    
	  filename = folder_name + '%05d.mat'%(t)
	  scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'albedo': mesh.albedo})
	  l2[t] = l2_record
	  albedo[t] = mesh.albedo


	filename = folder_name + 'progress.mat'
	scipy.io.savemat(filename, mdict={'albedo':albedo, 'l2': l2})

optimization(20)

