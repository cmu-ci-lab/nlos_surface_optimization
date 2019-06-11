import numpy as np
import time
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/ubuntu/install_dirs/libigl/python/')
sys.path.append('../smoothed_transient')
import math
import pyigl as igl
import scipy.io
import ggx
import renderer

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  alpha = .3

mesh = MESH()
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
mesh.v = np.array(v, dtype=np.float32, order = 'C')
mesh.f = np.array(gt_mesh.f, dtype=np.int32, order='C')



numBins = 1200
num_sample = 500000
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

tic = time.time()
renderer.renderStreamedTransient(origin, normal, mesh.v, mesh.f, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, 1, 1)
print(time.time() - tic)
filename = os.getcwd() + '/transient_no_smoothing_old.mat'
scipy.io.savemat(filename, mdict={'transient':transient})


tic = time.time()
ggx.renderStreamedTransient(origin, normal, mesh.v, mesh.f, mesh.alpha, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, 10, 1)
print(time.time() - tic)

tic = time.time()
ggx.renderStreamedTransient(origin, normal, mesh.v, mesh.f, mesh.alpha, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, 10, 1)
print(time.time() - tic)
filename = os.getcwd() + '/transient_smoothing.mat'
scipy.io.savemat(filename, mdict={'transient':transient})


tic = time.time()
ggx.renderStreamedGradient(origin, normal, mesh.v, mesh.f, mesh.alpha, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gradient, data, weight, 10, 1, 0)
print(time.time() - tic)

gt_transient = np.array(transient)

tic = time.time()
a = ggx.renderStreamedGradientAlpha(origin, normal, mesh.v, mesh.f, 0.2, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gt_transient, weight, 10, 1)
print(time.time() - tic)
print(a)
'''
#tic = time.time()
#renderer.renderTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths)
#print(time.time() - tic)

# dummy one to load the module
renderer.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, data, pathlengths,1, 1)
'''
'''
tic = time.time()
renderer.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths,1, 1)
print(time.time() - tic)

filename = os.getcwd() + '/transient_no_smoothing.mat'
scipy.io.savemat(filename, mdict={'transient':transient})

tic = time.time()
renderer.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths,10, 1)
print(time.time() - tic)

filename = os.getcwd() + '/transient_smoothing.mat'
scipy.io.savemat(filename, mdict={'transient':transient})


filename = os.getcwd() + '/transient_smoothing_large.mat'
scipy.io.savemat(filename, mdict={'transient':transient})
'''

#tic = time.time()
#ggx.renderStreamedGradient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gradient, data, weight, 10, 10)
#print(time.time() - tic)
