import numpy as np
import time
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import pyigl as igl
import scipy.io
import jitter

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

#mesh = MESH()
#z = .38 
#v = np.array([[-.25, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [0, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,.7]])
#f = np.array([[0, 1, 2], [0,2,3]])
#mesh.v = np.array(v, dtype=np.float32, order = 'C')
#mesh.f = np.array(f, dtype=np.int32, order = 'C')       

bunny_location = os.getcwd() + '/../../mesh_processing/data/bunny_parallel_T.obj'
gt_mesh = MESH()
igl.readOBJ(bunny_location, gt_mesh.v, gt_mesh.f)

v = np.array(gt_mesh.v.leftCols(3)/100, dtype=np.float32, order='C')
v[:,2] = v[:,2] - 0.5
faces = np.array(gt_mesh.f, dtype=np.int32, order='C')


numBins = 1200
num_sample = 50000
lower_bound = 0
resolution = 0.0012
upper_bound = numBins*resolution

transient = np.zeros((1,numBins), dtype=np.double, order = 'C')
data  = np.zeros((1,numBins), dtype=np.double, order = 'C')
weight  = np.ones((1,numBins), dtype=np.double, order = 'C')
pathlengths = np.zeros(numBins, dtype=np.double, order = 'C')
origin = np.array([[0,0,0]],dtype=np.float32, order='C')
normal = np.array([[0,0,1]],dtype=np.float32, order='C')
gradient = np.zeros(v.shape, dtype=np.double, order='C')
#tic = time.time()
#renderer.renderTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths)
#print(time.time() - tic)

# dummy one to load the module

jitter_weight = np.array([[1]], dtype=np.double, order='C')
jitter_offset = 0
jitter.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, data, pathlengths, jitter_weight, jitter_offset)
tic = time.time()
jitter.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, data, pathlengths, jitter_weight, jitter_offset)
print(time.time() - tic)
filename = os.getcwd() + '/transient.mat'
scipy.io.savemat(filename, mdict={'transient':data})

jitter_filename = os.getcwd() + '/jitter_info.mat'
jitter_info = scipy.io.loadmat(jitter_filename)
jitter_weight = np.array(jitter_info['jitter_weight'], dtype=np.double, order='C')
jitter_grad = np.array(jitter_info['jitter_grad'], dtype=np.double, order='C')
jitter_offset = jitter_info['jitter_offset']

tic = time.time()
jitter.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, data, pathlengths, jitter_weight, jitter_offset)
print(time.time() - tic)

filename = os.getcwd() + '/transient_w_jitter.mat'
scipy.io.savemat(filename, mdict={'transient':data})


jitter.renderStreamedGradient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, jitter_weight, jitter_grad, jitter_offset, transient, pathlengths, gradient, data, weight, 0) 


#tic = time.time()
#renderer.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths,10, 10)
#print(time.time() - tic)

#filename = os.getcwd() + '/transient_smoothing_large.mat'
#scipy.io.savemat(filename, mdict={'transient':transient})
'''
tic = time.time()
renderer.renderStreamedGradient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gradient, data, weight, 10, 1)
print(time.time() - tic)

tic = time.time()
renderer.renderStreamedGradient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gradient, data, weight, 10, 10)
print(time.time() - tic)
'''
