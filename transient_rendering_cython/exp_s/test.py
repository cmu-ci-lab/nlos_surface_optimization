import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.append('../exp_bunny')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time
import argparse
import rendering
import torch
from torch import optim
from adam_modified import Adam_Modified
import pyigl as igl


class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 2048 
  distance_resolution = 1.2*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  space_carving_projection = 0
  smooth_weight = 0.0001 
  smooth_ratio = .5
  gamma = 0
  resolution = 64
  loss_epsilon  = 10**-5
  T = 2
  bin_refine_resolution = 10
  sigma_bin = 1
  testing_flag = 1
  edge_lr_ratio = 1

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

def optimization(ratio):
	lr0 = 0.0001
	lr = lr0
	folder_name = os.getcwd() + '/progress-b-1-%f/'% ratio
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/transient.mat'
	setup = scipy.io.loadmat(filename)

	gt_transient = np.array(setup['rect_data'], dtype=np.double, order = 'C')
	gt_transient[:,:,range(600}] = 0
	gt_transient = np.reshape((gt_transient, opt.resolution*opt.resolution, 2048))

	opt = OPT(20000)
	mesh = MESH()

	mesh_init_location = os.getcwd() + '/cnlos_s_threshold.obj'
	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

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
	        rendering.removeTriangle(mesh,opt)
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
	    transient, grad, length = rendering.inverseRendering(mesh, gt_transient, weight, opt)	    
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
	    print('%05d update time: %8.8f L2 loss: %8.8f  old_l2 loss: %8.8f'% (t, time.time() - tic, l2, original_l2))
	    l2_record[t] = l2	
	    l2_original_record[t] = original_l2
	    v2_record[t] = v2
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'transient':transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'smoothing_grad':smoothing_grad, 'sample_num': opt.sample_num})
	    
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
	            continue

	    optimization_v.grad.data = torch.from_numpy(grad[mesh.v_edge==0,:]).float()
	    optimization_v_edge.grad.data = torch.from_numpy(grad[mesh.v_edge==1,:]).float()
	    optimizer.step()
	    mesh.v[mesh.v_edge==0,:] = np.array(optimization_v.data.numpy(), dtype=np.float32, order='C')
	    mesh.v[mesh.v_edge==1,:] = np.array(optimization_v_edge.data.numpy(), dtype=np.float32, order='C')

	    if run_count == 15:
	        remesh_flag = True
	
	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record, 'l2_original_record': l2_original_record, 'weight':weight})

optimization(100)

