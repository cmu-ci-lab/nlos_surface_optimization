import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
import math
import time

from itertools import compress
import pyigl as igl
import element_wise_manipulation
import rotation_matrix
import rendering_igl

class OPT:
  sample_num = 5000
  #max_distance_bin = 1146
  max_distance_bin = 2000
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  vn = igl.eigen.MatrixXd()
  doublearea = igl.eigen.MatrixXd()

mesh_location = '../mesh_processing/data/bunny.obj'

mesh = MESH()
#mesh.v = igl.eigen.MatrixXd([[-1,-1,0.9],[1, -1, 1],[1, 1, 1.2],[-1, 1, 1],[-2, -2, 1], [2,1, 1]])
#mesh.f = igl.eigen.MatrixXd([[0,2,1],[0, 3,2], [4,3,0], [1,2,5]]).castint()

read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)

igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
igl.per_vertex_normals(mesh.v, mesh.f, mesh.vn)
igl.doublearea(mesh.v, mesh.f, mesh.doublearea)

mesh.fn = np.array(mesh.fn)
mesh.vn = np.array(mesh.vn)
f = 0.5
z = -2

sensor = np.array([f, 0, z])
lighting = np.array([-f, 0, z])

sensor_normal = np.array([0, 0, 1])
lighting_normal = np.array([0, 0, 1])

opt = OPT()
opt.normal = 'n'
barycoord = rendering_igl.random_barycoord(mesh, opt.sample_num)
mesh_transient = rendering_igl.mesh_sampling(mesh,barycoord, lighting, sensor, lighting_normal, sensor_normal, opt)
#grad = rendering_igl.mesh_grad_sampling(mesh, barycoord, lighting, sensor, lighting_normal, sensor_normal, opt)

 
filename = os.getcwd() + '/python_mesh_grad.mat'
scipy.io.savemat(filename, mdict={'lighting':lighting, 'sensor':sensor, 'barycoord':barycoord, 'mesh_v':np.array(mesh.v), 'mesh_f':np.array(mesh.f), 'mesh_transient':mesh_transient})
#scipy.io.savemat(filename, mdict={'lighting':lighting, 'sensor':sensor, 'barycoord':barycoord, 'mesh_v':np.array(mesh.v), 'mesh_f':np.array(mesh.f), 'grad':grad, 'mesh_transient':mesh_transient})
