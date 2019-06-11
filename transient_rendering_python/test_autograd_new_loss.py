import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time


import pyigl as igl
import torch
from torch.autograd import Variable

import rotation_matrix
import rendering_igl
import rendering_gpu_igl
import mesh_intersection_grad_igl

class OPT:
  sample_num = 1000
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

#mesh_location = '../mesh_processing/data/bunny.obj'

mesh = MESH()
mesh.v = igl.eigen.MatrixXd([[-1,-1,0.9],[1, -1, 1],[1, 1, 1.2],[-1, 1, 1],[-2, -2, 1], [2,1, 1]])
mesh.f = igl.eigen.MatrixXd([[0,2,1],[0, 3,2], [4,3,0], [1,2,5]]).castint()
igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
mesh.fn = np.array(mesh.fn)

f = 0.5
z = 0

sensor = np.array([f, 0, z])
lighting = np.array([-f, 0, z])

sensor_normal = np.array([0, 0, 1])
lighting_normal = np.array([0, 0, 1])

opt = OPT()

phi = 2*math.pi*np.random.rand(opt.sample_num)
theta = np.arccos(np.random.rand(opt.sample_num))
R = np.zeros([3,3])
rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)
direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)

triangleIndexMap = mesh_intersection_grad_igl.find_triangle(mesh, direction, lighting, sensor)

transient = rendering_igl.angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt)



mesh.v = Variable(torch.from_numpy(np.array(mesh.v)).cuda(), requires_grad = True)
mesh.f = Variable(torch.from_numpy(np.array(mesh.f)).long().cuda())

tic = time.time()
angular_transient = rendering_gpu_igl.angular_sampling(mesh, direction, triangleIndexMap, lighting, sensor, lighting_normal, sensor_normal, opt)
print(' igl forward')
print(time.time() - tic)

#weight = Variable(torch.ones(opt.max_distance_bin).double().cuda())
weight = Variable(torch.from_numpy(transient).cuda())

tic = time.time()
loss = torch.mul(weight, angular_transient).sum()
loss.backward()
grad = mesh.v.grad
print(' gradient')
print(time.time() - tic)

filename = os.getcwd() + '/python_test_grad.mat'
scipy.io.savemat(filename, mdict={'v':mesh.v.cpu().data.numpy(), 'f':mesh.f.cpu().data.numpy(),'lighting':lighting, 'direction':direction, 'angular_transient':angular_transient.cpu().data.numpy(),'grad':grad.cpu().data.numpy()})

