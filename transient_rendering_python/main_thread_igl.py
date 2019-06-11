import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/chiayint/research/libigl/python/')

import math
import time
import threading
import queue

import pyigl as igl
import rotation_matrix

import rendering_igl
import rendering_grad
import rendering_gpu_igl
import mesh_intersection_grad_igl

import torch
from torch.autograd import Variable

def run_forward(mesh, opt):
	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	 
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal)

	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)

	angular_transient = rendering_igl.angular_sampling(mesh, direction, opt.lighting, opt.sensor, opt.lighting_normal, opt.sensor_normal, opt)
	return angular_transient

def run_gradient_est(mesh, opt, weight): 
	new_mesh = type('',(),{})()
	new_mesh.v = Variable(torch.from_numpy(np.array(mesh.v)).cuda(), requires_grad=True)

	new_mesh.f = Variable(torch.from_numpy(np.array(mesh.f)).long().cuda(), requires_grad = False)

	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	 
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal)

	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
	triangleIndexMap = mesh_intersection_grad_igl.find_triangle(mesh, direction, opt.lighting, opt.sensor)

	angular_transient = rendering_gpu_igl.angular_sampling(new_mesh, direction, triangleIndexMap, opt.lighting, opt.sensor, opt.lighting_normal, opt.sensor_normal, opt)
	angular_transient.backward(weight.cuda())
	grad = new_mesh.v.grad.data.cpu()
	return grad

def worker(tid, mesh, opt, weight, queue, isGPU=True):
    with torch.cuda.device(tid):
        tic = time.time()
        gradient = run_gradient_est(mesh, opt, weight)  
        
        print('GPU {} start time {} elapse {}'.format(tid, tic, time.time() - tic))
        queue.put(gradient)


class OPT:
  sample_num = 2500
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  f = 0.5
  z = 1.8
  sensor = np.array([f, 0, z])
  lighting = np.array([-f, 0, z])
  sensor_normal = np.array([0, 0, -1])
  lighting_normal = np.array([0, 0, -1])

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()


mesh_location = '../mesh_processing/data/bunny.obj'
mesh = MESH()
read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)

igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
mesh.fn = np.array(mesh.fn)

opt = OPT()

gpu_IDs = [2, 3]*3
gpu_threads = []

render_opt = OPT()
render_opt.sample_num = 200000

tic = time.time()
weight = torch.DoubleTensor(torch.from_numpy(run_forward(mesh, render_opt)))
print('forward')
print(time.time()- tic)
queue = queue.Queue()

tic = time.time()
for tid in gpu_IDs:
    t = threading.Thread(target=worker, args=[tid, mesh, opt, weight, queue, True])
    print("start GPU thread {0}...".format(tid))
    time.sleep(0.2)
    t.daemon = True
    t.start()
    gpu_threads.append(t)

for t in gpu_threads: t.join()

grad = torch.DoubleTensor(mesh.v.rows(), mesh.v.cols()).zero_()
cnt = 1
while not queue.empty():
    print(cnt)
    grad += queue.get()
    cnt += 1

print("=========ALL DONE============")
print(time.time() - tic)
