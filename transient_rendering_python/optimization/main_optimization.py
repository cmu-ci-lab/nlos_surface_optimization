import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time

import rotation_matrix
import rendering
import torch
from torch import optim

import pyigl as igl

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
 
  measurement_num = lighting.shape[0] 
  sensor = np.array([0, 0, 0.0])
  sensor = np.tile(sensor, (measurement_num,1))
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

#device = torch.device('cuda:1')
device = torch.device('cpu')
folder_name = os.getcwd() + '/progress/'
if not os.path.isdir(folder_name):
  os.mkdir(os.getcwd() + '/progress/')


filename = os.getcwd() + '/setup.mat'
setup = scipy.io.loadmat(filename)

gt_transient = setup['gt_transient']
#mesh_location = '../mesh_processing/data/bunny.obj'
gt_mesh = MESH()
gt_mesh.v = igl.eigen.MatrixXd(torch.from_numpy(setup['gt_v']).numpy())
gt_mesh.f = igl.eigen.MatrixXi(torch.from_numpy(setup['gt_f']).numpy())
igl.per_face_normals(gt_mesh.v, gt_mesh.f, gt_mesh.fn)

opt = OPT(5000)
render_opt = OPT(100000)

mesh = MESH()
mesh.v = igl.eigen.MatrixXd(torch.from_numpy(setup['v']).numpy())
mesh.f = gt_mesh.f
igl.per_face_normals(mesh.v, mesh.f, mesh.fn)

mesh_optimization = MESH()
mesh_optimization.v = torch.from_numpy(np.array(mesh.v)).to(device)
mesh_optimization.v.requires_grad_()
mesh_optimization.f = torch.from_numpy(np.array(mesh.f)).long().to(device)

l2, transient = rendering.evaluate_L2(gt_transient, mesh, render_opt)
v2 = rendering.evaluate_vertex_L2(gt_mesh, mesh)
print('%05d update time: %5.5f L2 loss: %5.5f vertex L2 loss: %5.5f'% (0, 0, l2, v2))

filename = os.getcwd() + '/progress/init.mat'
scipy.io.savemat(filename, mdict={'f': np.array(mesh.f), 'v':np.array(mesh.v), 'optim_v':mesh_optimization.v.cpu().data.numpy(), 'gt_v':np.array(gt_mesh.v), 'transient':transient, 'l2': l2, 'v2':v2, 'gt_transient': gt_transient})

optimizer = optim.Adam([mesh_optimization.v], lr=0.0005)

T = 200

l2_record = np.empty(T)	
v2_record = np.empty(T)	

for t in range(T):
	tic = time.time()
	for index in np.random.permutation(opt.lighting.shape[0]):
		optimizer.zero_grad()
	
		loss = rendering.loss_func_parallel(index, gt_transient, mesh, mesh_optimization, opt, render_opt, device)
		loss.backward()
		optimizer.step()
        
		mesh.v = igl.eigen.MatrixXd(mesh_optimization.v.cpu().data.numpy())
		igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
	

	l2, transient = rendering.evaluate_L2(gt_transient, mesh, render_opt)
	v2 = rendering.evaluate_vertex_L2(gt_mesh, mesh)
	print('%05d update time: %5.5f L2 loss: %5.5f vertex L2 loss: %5.5f'% (t, time.time()- tic, l2, v2))
	l2_record[t] = l2	
	v2_record[t] = v2	
	filename = os.getcwd() + '/progress/%05d.mat'%(t)
	scipy.io.savemat(filename, mdict={'v':np.array(mesh.v),  'transient':transient, 'l2':l2, 'v2':v2})

filename = os.getcwd() + '/progress/loss_val.mat'
scipy.io.savemat(filename, mdict={'l2':l2_record,  'v2':v2_record})
