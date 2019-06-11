import numpy as np
import time
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import pyigl as igl

import renderer

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

mesh_location = '../../mesh_processing/data/bunny.obj'
#mesh_location = '../../mesh_processing/data/bunny_parallel_T.obj'
mesh = MESH()
read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)

v = np.array(mesh.v.block(0,0,mesh.v.rows(),3), dtype = np.float32, order='C')


faces = np.array(mesh.f, dtype = np.int32, order = 'C')
num_sample = 5000000
lower_bound = 0
upper_bound = 300
resolution = 0.12
numBins = math.ceil((upper_bound - lower_bound)/resolution)

transient = np.zeros(numBins, dtype=np.double, order = 'C')
pathlengths = np.zeros(numBins, dtype=np.double, order = 'C')
origin = np.array([0,0,0],dtype=np.float32, order='C')
normal = np.array([0,0,1],dtype=np.float32, order='C')
gradient = np.zeros((numBins,3*v.shape[0]), dtype=np.double, order='C')
tic = time.time()
renderer.renderTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths)
print(time.time() - tic)

tic = time.time()
renderer.renderStreamedTransient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths)
print(time.time() - tic)

tic = time.time()
renderer.renderStreamedGradient(origin, normal, v, faces, num_sample, lower_bound, upper_bound, resolution, transient, pathlengths, gradient)
print(time.time() - tic)
