import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time
import argparse
from itertools import compress

import rendering
import torch
from torch import optim

import pyigl as igl

class SMOOTH_OPT:
  v_shape = (7,7)
  weight = 0.00001 

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  space_carving_projection = 1
  thread_num = 20
  w_width = 100
  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-1, 1, 7), np.linspace(-1, 1, 7))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
 
  measurement_num = lighting.shape[0] 
  sensor = lighting
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  doublearea = igl.eigen.MatrixXd()

def optimization(lr):
	folder_name = os.getcwd() + '/progress3-%f/'%lr
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/setup_4.mat'
	setup = scipy.io.loadmat(filename)

	gt_transient = setup['gt_transient']
	gt_mesh = MESH()
	gt_mesh.v = igl.eigen.MatrixXd(torch.from_numpy(setup['gt_v']).numpy())
	gt_mesh.f = igl.eigen.MatrixXi(torch.from_numpy(setup['gt_f']).numpy())
	igl.per_face_normals(gt_mesh.v, gt_mesh.f, gt_mesh.fn)
	igl.doublearea(gt_mesh.v, gt_mesh.f, gt_mesh.doublearea)

	opt = OPT(5000)
	render_opt = OPT(50000)
	smooth_opt = SMOOTH_OPT()
	opt.space_carving_projection = 1

	space_carving_location = os.getcwd() + '/space_carving_mesh4.obj'
	space_carving_mesh = MESH()
	igl.readOBJ(space_carving_location, space_carving_mesh.v, space_carving_mesh.f)

	mesh = MESH()
	mesh_init_location = os.getcwd() + '/cnlos_5.obj'
	igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	#rendering.space_carving_initialization(mesh, space_carving_mesh, opt)

	igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
	igl.doublearea(mesh.v, mesh.f, mesh.doublearea)

	mesh_optimization = MESH()
	mesh_optimization.v = torch.from_numpy(np.array(mesh.v))
	mesh_optimization.v.requires_grad_()

	l2, transient, original_l2 = rendering.evaluate_smooth_L2_collocate(gt_transient, mesh, render_opt, smooth_opt)
	print('%05d update time: %5.5f L2 loss: %5.5f old L2 loss: %5.5f'% (0, 0, l2, original_l2))

	filename = folder_name + 'init.mat'
	scipy.io.savemat(filename, mdict={'f': np.array(mesh.f), 'v':np.array(mesh.v), 'optim_v':mesh_optimization.v.data.numpy(), 'gt_v':np.array(gt_mesh.v), 'transient':transient, 'l2': l2, 'gt_transient': gt_transient})

	optimizer = optim.Adam([mesh_optimization.v], lr = lr)

	dummy_loss = torch.sum(mesh_optimization.v)
	dummy_loss.backward()

	T = 10

	l2_record = np.empty(T)	

	for t in range(T):
	    if t%30 == 0:
	        if opt.w_width >= 10:
	            opt.w_width -= 10
	            render_opt.w_width -= 10
	        opt.sample_num += 500
	        render_opt.sample_num += 5000    
	    
	    tic = time.time()
	    optimizer.zero_grad()
	    
	    #grad = np.zeros((mesh.v.rows(),3))
	    #for index in range(opt.lighting.shape[0]):
	    #    grad += rendering.grad_collocate(index, gt_transient, transient, mesh, opt)
	    grad = rendering.grad_parallel(gt_transient, transient, mesh, opt)
	    grad += rendering.smooth_grad(mesh, smooth_opt)
	    mesh_optimization.v.grad.data = torch.from_numpy(grad)
	    optimizer.step()
	    
	    if opt.space_carving_projection == 1:
	      mesh.v = rendering.space_carving_projection(mesh_optimization, space_carving_mesh)
	    else:
	      mesh.v = igl.eigen.MatrixXd(mesh_optimization.v.data.numpy())
  
	    igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
	    igl.doublearea(mesh.v, mesh.f, mesh.doublearea)
	      
	    l2, transient, original_l2 = rendering.evaluate_smooth_L2_collocate(gt_transient, mesh, render_opt, smooth_opt)
	    print('%05d update time: %5.5f L2 loss: %5.5f old L2 loss: %5.5f'% (t, time.time() - tic, l2, original_l2))
	    l2_record[t] = l2	
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':np.array(mesh.v),  'transient':transient, 'l2':l2, 'origin_v': mesh_optimization.v.data.numpy(), 'grad': mesh_optimization.v.grad.data.numpy(), 'w_width': opt.w_width})
	    mesh_optimization.v.data = torch.from_numpy(np.array(mesh.v))

	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record})
