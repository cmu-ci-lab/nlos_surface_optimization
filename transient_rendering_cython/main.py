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
  weight = 0.001 
  #weight = 0 

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  space_carving_projection = 1
  w_width = 50
  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))
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



def optimization(lr):
	folder_name = os.getcwd() + '/progress4-%f/'%lr
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)

	filename = os.getcwd() + '/setup/plane.mat'
	setup = scipy.io.loadmat(filename)

	gt_transient = np.array(setup['gt_transient'], dtype=np.double, order = 'C')
	gt_mesh = MESH()
	gt_mesh.v = setup['gt_v']
	gt_mesh.f = setup['gt_f']
        #gt_mesh.fn, gt_mesh.face_area = rendering.face_normal_and_area(gt_mesh.v, gt_mesh.f)	

	opt = OPT(500000)
	smooth_opt = SMOOTH_OPT()

	space_carving_location = os.getcwd() + '/space_carving/plane.obj'
	space_carving_mesh = MESH()
	igl.readOBJ(space_carving_location, space_carving_mesh.v, space_carving_mesh.f)
	space_carving_mesh.v = np.array(space_carving_mesh.v, dtype=np.float32, order='C') 
	space_carving_mesh.f = np.array(space_carving_mesh.f, dtype=np.int32, order='C') 


	mesh = MESH()
	#mesh_init_location = os.getcwd() + '/cnlos_mesh/1.obj'
	#igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	mesh.v = np.array(gt_mesh.v + np.array([0,0,0.1]) , dtype=np.float32, order = 'C')
	#mesh.v = np.array(gt_mesh.v, dtype=np.float32, order = 'C')
	mesh.f = np.array(gt_mesh.f, dtype=np.int32, order = 'C')       
	#mesh.fn, mesh.face_area = rendering.face_normal_and_area(mesh.v, mesh.f)	
	mesh_optimization = MESH()
	mesh_optimization.v = torch.from_numpy(mesh.v)
	mesh_optimization.v.requires_grad_()

	optimizer = optim.Adam([mesh_optimization.v], lr = lr)

	dummy_loss = torch.sum(mesh_optimization.v)
	dummy_loss.backward()

	T = 500

	l2_record = np.empty(T)	
	l2_original_record = np.empty(T)	

	for t in range(T):
	    if t%50 == 0:
	        if opt.w_width >= 2:
	            opt.w_width -= 2
	    print('t = %d'% t)    
	    tic = time.time()
	    transient, grad, pathlength = rendering.inverseRendering(mesh, gt_transient, opt)	    
	    grad = np.sum(grad, axis=0)/opt.lighting.shape[0]
	    grad = np.reshape(grad, mesh.v.shape)
	    grad[:,range(2)] = 0
	    grad += rendering.smooth_grad(mesh, smooth_opt)

	    l2, original_l2 = rendering.evaluate_loss(gt_transient, transient, mesh, opt,  smooth_opt)  
	    print('%05d update time: %5.5f L2 loss: %5.5f old L2 loss: %5.5f'% (t, time.time() - tic, l2, original_l2))
	    l2_record[t] = l2	
	    l2_original_record[t] = original_l2
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'transient':transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'w_width': opt.w_width})

	    mesh_optimization.v.grad.data = torch.from_numpy(grad).float()
	    optimizer.step()
	    mesh.v = np.array(mesh_optimization.v.data.numpy(), dtype=np.float32, order='C')
	    if opt.space_carving_projection == 1:
	       rendering.space_carving_projection(mesh.v, space_carving_mesh)
  
	    #igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
	    #igl.doublearea(mesh.v, mesh.f, mesh.doublearea)
	    
	    mesh_optimization.v.data = torch.from_numpy(mesh.v)

	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record})


optimization(1)
