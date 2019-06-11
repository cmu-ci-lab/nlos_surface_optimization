import numpy as np
import scipy.io
import sys, os
import math
import time


import trimesh
import torch
from torch.autograd import Variable

import rendering_gpu
import rotation_matrix

class OPT:
  sample_num = 2500
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'


mesh_location = '../mesh_processing/data/bunny.obj'
mesh = trimesh.load(mesh_location)

tic = time.time()

with torch.cuda.device(2):
       
	v = torch.from_numpy(np.array(mesh.vertices)).cuda()
	print(time.time() - tic)
	mesh.v = Variable(v, requires_grad=True)
	mesh.f = Variable(torch.from_numpy(np.array(mesh.faces)).long().cuda(), requires_grad = False)
	mesh.fn = Variable(torch.from_numpy(np.array(mesh.face_normals)).cuda())


	f = 0.5
	z = 1.8

	sensor = np.array([f, 0, z])
	lighting = np.array([-f, 0, z])

	sensor_normal = np.array([0, 0, -1])
	lighting_normal = np.array([0, 0, -1])

	opt = OPT()

	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	 
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)

	 
	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)


	angular_transient = rendering_gpu.angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt)
	print(time.time() - tic)

	angular_transient.backward(torch.ones_like(angular_transient))
	grad = mesh.v.grad

	print(time.time() - tic)
	angular_transient = angular_transient.cpu()
	grad = grad.cpu()

filename = os.getcwd() + '/python_test.mat'
scipy.io.savemat(filename, mdict={'lighting':lighting, 'direction':direction, 'angular_transient':angular_transient.data.numpy()})

