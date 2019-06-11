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

  lighting = np.array([-1.0, -1.0, 0.0])
 
  measurement_num = lighting.shape[0] 
  sensor = np.array([0, 0, 0.0])
  sensor_normal = np.array([0, 0, 1])
  
  lighting_normal = np.array([0, 0, 1])

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

device = torch.device('cpu')
folder_name = os.getcwd() + '/grad-progress2/'
if not os.path.isdir(folder_name):
  os.mkdir(folder_name)



mesh = MESH()
mesh.v = igl.eigen.MatrixXd([[-1,-1,0.9],[1, -1, 1],[1, 1, 1.2],[-1, 1, 1],[-2, -2, 1], [2,1, 1]])
mesh.f = igl.eigen.MatrixXd([[0,2,1],[0, 3,2], [4,3,0], [1,2,5]]).castint()

mesh_optimization = MESH()
mesh_optimization.v = torch.from_numpy(np.array(mesh.v)).to(device)
mesh_optimization.v.requires_grad_()
mesh_optimization.f = torch.from_numpy(np.array(mesh.f)).long().to(device)

opt = OPT(5000)
T = 50000
#bin_num = 780
filename = folder_name  + 'setup.mat'
scipy.io.savemat(filename, mdict={'bin_width': opt.distance_resolution, 'lighting': opt.lighting, 'sensor':opt.sensor, 'v':np.array(mesh.v),  'f': np.array(mesh.f), 'T':T})


grad = np.zeros((mesh.v.rows(), 3))
for t in range(T):
    print(t) 
    phi = 2*math.pi*np.random.rand(opt.sample_num)
    theta = np.arccos(np.random.rand(opt.sample_num))
    R = np.zeros([3,3])
    rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal)
    direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
    triangleIndexMap = rendering.find_triangle(mesh, direction, opt.lighting, opt.sensor)


    transient_differentiable = rendering.render_differentiable(mesh_optimization, direction, triangleIndexMap, opt.lighting, opt.sensor, opt.lighting_normal, opt.sensor_normal, opt, device)

    transient_differentiable.backward(torch.ones_like(transient_differentiable))
    grad += mesh_optimization.v.grad.data.numpy()
    mesh_optimization.v.grad.data.zero_()

    #filename = folder_name + '%05d'%(t)
    #scipy.io.savemat(filename, mdict={'v':mesh_optimization.v.data.numpy(), 'f':mesh_optimization.f.data.numpy(),'lighting':opt.lighting, 'direction':direction, 'angular_transient':transient_differentiable.data.numpy(),'grad':grad[t,:,:]})
	

grad /= T
filename = folder_name + 'grad.mat'
scipy.io.savemat(filename, mdict={'grad':grad})

