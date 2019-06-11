import numpy as np
import scipy.io
import sys, os
import math
import time
import threading
import queue

import trimesh
import torch
from torch.autograd import Variable

import rendering_gpu
import rotation_matrix

def run_gradient_est(mesh, opt, weight): 
	new_mesh = type('',(),{})()
	new_mesh.v = Variable(torch.from_numpy(np.array(mesh.vertices)).cuda(), requires_grad=True)
	new_weight = Variable(weight.cuda(), requires_grad = False)

	new_mesh.f = Variable(torch.from_numpy(np.array(mesh.faces)).long().cuda(), requires_grad = False)
	new_mesh.fn = Variable(torch.from_numpy(np.array(mesh.face_normals)).cuda())

	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	 
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal)

	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)

	angular_transient = rendering_gpu.angular_sampling(new_mesh, direction, opt.lighting, opt.sensor, opt.lighting_normal, opt.sensor_normal, opt)

	angular_transient.backward(new_weight)
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

mesh_location = '../mesh_processing/data/bunny.obj'
mesh = trimesh.load(mesh_location)

opt = OPT()

gpu_IDs = [3,3,3]
gpu_threads = []

weight = torch.DoubleTensor(opt.max_distance_bin).fill_(1)
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

grad = torch.DoubleTensor(mesh.vertices.shape[0], mesh.vertices.shape[1]).zero_()
cnt = 1
while not queue.empty():
    print(cnt)
    grad += queue.get()
    cnt += 1

print("=========ALL DONE============")
print(time.time() - tic)
