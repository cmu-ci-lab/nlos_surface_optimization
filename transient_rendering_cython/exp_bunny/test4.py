import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')
import math
import time
import copy
import argparse
import rendering
import torch
from torch import optim
from adam_modified import Adam_Modified
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
  smooth_weight = 0.0001 
  smooth_ratio = .5
  gamma = 0
  resolution = 64
  loss_epsilon  = 10**-5
  T = 500
  bin_refine_resolution = 10
  sigma_bin = 1
  testing_flag = 1
  edge_lr_ratio = 1


class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  f_affinity = 0
  v_edge = 0

def optimization(ratio):
	lr0 = 0.0001/3
	lr = lr0
	folder_name = os.getcwd() + '/progress-b-4-%f/'% ratio
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	resolution = 128 
	batch = 4

	opt = OPT(20000, resolution)
	a = np.arange(opt.lighting.shape[0])
	b = np.split(a, batch)

	gt_transient = np.empty((opt.resolution*opt.resolution, opt.max_distance_bin), dtype=np.double, order='C')
	for i, x in zip(range(len(b)), b):
	  filename = os.getcwd() + '/setup/bunny_transient_%d_%d.mat'%(resolution,i)
	  setup = scipy.io.loadmat(filename)

	  gt_transient[x,:] = np.array(setup['gt_transient'], dtype=np.double, order = 'C')


	gt_mesh = MESH()
	gt_mesh.v = igl.eigen.MatrixXd(np.array(setup['gt_v'], dtype = np.double))
	gt_mesh.f = igl.eigen.MatrixXd(np.array(setup['gt_f'], dtype = np.double)).castint()

	opt.smooth_weight = 0.001
	mesh = MESH()

	mesh_init_location = os.getcwd() + '/init/cnlos_bunny_threshold.obj'
	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

	opt.smooth_ratio = ratio
      
	rendering.isotropic_remeshing(mesh, .5/opt.resolution)
	old_v = np.array(mesh.v)
	rendering.compute_mesh_affinity(mesh)
	rendering.border_indicator(mesh)


	optimization_v = torch.from_numpy(mesh.v[mesh.v_edge==0,:])
	optimization_v.requires_grad_()
	optimization_v_edge = torch.from_numpy(mesh.v[mesh.v_edge==1,:])
	optimization_v_edge.requires_grad_()

	weight = rendering.create_weighting_function(gt_transient, opt.gamma)
	
	optimizer = Adam_Modified([{'params':optimization_v}, {'params': optimization_v_edge, 'lr': lr*opt.edge_lr_ratio}], lr = lr)
	dummy_loss = torch.sum(optimization_v) + torch.sum(optimization_v_edge)
	dummy_loss.backward()
	scale_flag = False
	remesh_flag = False
	weight_flag = True
	run_count = 0

	l2_record = np.empty(opt.T)	
	v2_record = np.empty(opt.T)	
	l2_original_record = np.empty(opt.T)	

	for t in range(opt.T):
	    if remesh_flag:
	        print('remesh')
	        if mesh.f.shape[0] > 250000:
	            l2_record = l2_record[range(t)]
	            l2_original_record = l2_original_record[range(t)]
	            v2_record = v2_record[range(t)]
	            break

	        eltopo_tic = time.time()
	        rendering.el_topo_gradient(mesh, old_v)
	        mesh.v = np.array(mesh.v, dtype=np.double, order='C')
	        rendering.el_topo_remeshing(mesh, .5/opt.resolution)
	        rendering.isotropic_remeshing(mesh, .5/opt.resolution)

	        #rendering.compute_mesh_affinity(mesh)
	        #rendering.removeTriangle(mesh,opt)
	        #rendering.keep_largest_connected_component(mesh)
	        old_v = np.array(mesh.v)

	        rendering.compute_mesh_affinity(mesh)
	        rendering.border_indicator(mesh)
	        optimization_v = torch.from_numpy(mesh.v[mesh.v_edge==0,:])
	        optimization_v.requires_grad_()
	        optimization_v_edge = torch.from_numpy(mesh.v[mesh.v_edge==1,:])
	        optimization_v_edge.requires_grad_()

	        weight = rendering.create_weighting_function(gt_transient, opt.gamma)

	        optimizer = Adam_Modified([{'params':optimization_v}, {'params': optimization_v_edge, 'lr': lr*opt.edge_lr_ratio}], lr = lr)
	        dummy_loss = torch.sum(optimization_v) + torch.sum(optimization_v_edge)
	        dummy_loss.backward()

	        remesh_flag = False
	        run_count = 0

	    print('t = %d'% t)    
	    tic = time.time()


	    transient = np.zeros(gt_transient.shape, dtype=np.double, order = 'C')
	    grad = np.zeros(mesh.v.shape, dtype = np.double, order = 'C') 
	    tmp_opt = copy.deepcopy(opt)

	    for i, x in zip(range(len(b)),b):
	        tmp_opt.lighting = opt.lighting[x,:]
	        tmp_opt.lighting_normal = opt.lighting_normal[x,:]
 
	        transient[x,:], grad_tmp, length = rendering.inverseRendering(mesh, gt_transient[x,:], weight[x,:], tmp_opt)	    
	        grad += grad_tmp
	    grad /= len(b)
	    smoothing_val, smoothing_grad = rendering.renderStreamedNormalSmoothing(mesh)
	    l2,  original_l2 = rendering.evaluate_loss_with_normal_smoothness(gt_transient, weight, transient, smoothing_val, mesh, opt)  

	    if weight_flag:
	      opt.smooth_weight = original_l2/smoothing_val/opt.smooth_ratio
	      weight_flag = False
	      print('new smooth weight %f' %opt.smooth_weight)
	      if t > 0:
	        lr = (original_l2/l2_original_record[0]) * lr0 * ((0.99)**(t/15))
	        print('new lr %f' % lr)

	    grad += opt.smooth_weight * smoothing_grad
	    v2 = rendering.compute_v2(mesh.v, gt_mesh)
	    print('%05d update time: %8.8f L2 loss: %8.8f  old_l2 loss: %8.8f v2: %8.8f'% (t, time.time() - tic, l2, original_l2, v2))
	    l2_record[t] = l2	
	    l2_original_record[t] = original_l2
	    v2_record[t] = v2
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'smoothing_grad':smoothing_grad, 'sample_num': opt.sample_num})
	    #scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'transient':transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'smoothing_grad':smoothing_grad, 'sample_num': opt.sample_num})
	    
	    run_count += 1

	    if run_count > 2:
	        if ((l2_original_record[t-1] - original_l2)/l2_original_record[t-1])< opt.loss_epsilon or ((l2_record[t-1] - l2)/l2_record[t-1])< opt.loss_epsilon:
	            if opt.testing_flag == 1:
	                opt.testing_flag = 0
	                opt.smooth_ratio = 10 + t/100 
	                print('shading based')
	            else:
	                opt.testing_flag = 1
	                opt.resolution *= 1.5
	                opt.sample_num *= 1.5
	                opt.loss_epsilon /= 2
	                opt.smooth_ratio = ratio + t/10
	                print('remesh %d'%opt.resolution)

	            remesh_flag = True
	            weight_flag = True
	            #opt.gamma 
	            #weight = rendering.create_weighting_function(gt_transient, opt.gamma)
	            #opt.smooth_ratio *= 1.5
	            continue

	    optimization_v.grad.data = torch.from_numpy(grad[mesh.v_edge==0,:]).float()
	    optimization_v_edge.grad.data = torch.from_numpy(grad[mesh.v_edge==1,:]).float()
	    optimizer.step()
	    mesh.v[mesh.v_edge==0,:] = np.array(optimization_v.data.numpy(), dtype=np.float32, order='C')
	    mesh.v[mesh.v_edge==1,:] = np.array(optimization_v_edge.data.numpy(), dtype=np.float32, order='C')

	    if run_count == 15:
	        remesh_flag = True
	
	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record, 'l2_original_record': l2_original_record, 'v2_record':v2_record, 'weight':weight})

optimization(100)

