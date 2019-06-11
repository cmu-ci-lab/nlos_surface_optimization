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
  weight = 0 

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1200
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


lr = 0.01
filename = os.getcwd() + '/setup/bunny_transient.mat'
setup = scipy.io.loadmat(filename)
gt_transient = np.array(setup['gt_transient'], dtype=np.double, order = 'C')
gt_mesh = MESH()
gt_mesh.v = igl.eigen.MatrixXd(np.array(setup['gt_v'], dtype = np.double))
gt_mesh.f = igl.eigen.MatrixXd(np.array(setup['gt_f'], dtype = np.double)).castint()
		
opt = OPT(50000)
opt.max_distance_bin = gt_transient.shape[1]
smooth_opt = SMOOTH_OPT()
		
resolution = 64
T = 50
loss_epsilon = 10**-4
		


exp_folder = os.getcwd() + '/weight_test2/'
if not os.path.isdir(exp_folder):
  os.mkdir(exp_folder)

#gamma_list = [-2,-0.5,0, 1, 2.5, 5, 7.5, 10]
gamma_list = [-2, -0.5, 0, 5, 10]

for gamma in gamma_list:
	print('%f'%(gamma))
	folder_name = exp_folder + 'progress_%f/'%(gamma)
	if not os.path.isdir(folder_name):
	  os.mkdir(folder_name)
	#else:
	#  continue

	weight = rendering.create_weighting_function(gt_transient, gamma, 'i')

	mesh = MESH()
	z = .38 
	v = np.array([[-.25, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [0, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,z]])
	f = np.array([[0, 8,4], [0,7,8], [4,8,1],[1,8,5],[8,2,5],[8,6,2],[7,3,8],[3,6,8]])
	mesh.v = np.array(v, dtype=np.float32, order = 'C')
	mesh.f = np.array(f, dtype=np.int32, order = 'C')       
	#mesh_init_location = os.getcwd() + '/cnlos_bunny_threshold.obj'
	#igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
	#mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
	#mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

	mesh_optimization = MESH()
	mesh_optimization.v = torch.from_numpy(mesh.v)
	mesh_optimization.v.requires_grad_()

	optimizer = optim.Adam([mesh_optimization.v], lr = lr)
	dummy_loss = torch.sum(mesh_optimization.v)
	dummy_loss.backward()

	l2_record = np.zeros(T)	
	v2_record = np.zeros(T)	
	l2_original_record = np.zeros(T)	
	run_count = 0

	for t in range(T):

	    print('t = %d'% t)    
	    tic = time.time()
	    transient, grad, pathlength = rendering.inverseRendering(mesh, gt_transient, weight, opt)	    
	    #grad += smooth_opt.weight*rendering.renderStreamedCurvatureGradient(mesh)
	    l2,  original_l2, intensity_loss = rendering.evaluate_loss_with_curvature(gt_transient, weight, transient, mesh, opt,  smooth_opt)  
	    v2 = rendering.compute_v2(mesh.v, gt_mesh)
	    print('%05d update time: %8.8f L2 loss: %8.8f  old_l2 loss: %8.8f v2: %8.8f'% (t, time.time() - tic, l2, original_l2, v2))
	    l2_record[t] = l2	
	    l2_original_record[t] = original_l2
	    v2_record[t] = v2
	    filename = folder_name + '%05d.mat'%(t)
	    scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'transient':transient, 'l2':l2, 'l2_original':original_l2, 'grad': grad, 'w_width': opt.w_width, 'sample_num': opt.sample_num})

	    mesh_optimization.v.grad.data = torch.from_numpy(grad).float()
	    optimizer.step()
	    mesh.v = np.array(mesh_optimization.v.data.numpy(), dtype=np.float32, order='C')
	  
	    mesh_optimization.v.data = torch.from_numpy(mesh.v)
	    run_count += 1
	    if run_count > 2:
	        if ((l2_record[t-1] - l2)/l2_record[t-1])< loss_epsilon:
	            l2_record = l2_record[range(t)]
	            v2_record = v2_record[range(t)]
	            l2_original_record = l2_original_record[range(t)]
	            break

	filename = folder_name + 'loss_val.mat'
	scipy.io.savemat(filename, mdict={'l2':l2_record, 'l2_original_record': l2_original_record, 'v2_record':v2_record, 'weight':weight})

