import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time

import pyigl as igl

import rendering
import rotation_matrix

class OPT:
  sample_num = 500
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

class MESH:
  v = igl.eigen.MatrixXd()
  n = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

mesh_location = '../mesh_processing/data/bunny.obj'

mesh = MESH()
read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)

igl.per_face_normals(mesh.v, mesh.f, mesh.fn)

mesh.v = np.array(mesh.v)
mesh.f = np.array(mesh.f)
mesh.fn = np.array(mesh.fn)


f = 0.5
z = 1.8

sensor = np.array([f, 0, z])
lighting = np.array([-f, 0, z])

sensor_normal = np.array([0, 0, -1])
lighting_normal = np.array([0, 0, -1])

opt = OPT()
rounds = 10



phi = 2*math.pi*np.random.rand(opt.sample_num)
theta = np.arccos(np.random.rand(opt.sample_num))
 
 
R = np.zeros([3,3])
rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)

 
direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)

tic = time.time()
angular_transient = rendering.angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt)
print(time.time() - tic)

filename = os.getcwd() + '/python_test.mat'
scipy.io.savemat(filename, mdict={'lighting':lighting, 'direction':direction, 'angular_transient':angular_transient})

