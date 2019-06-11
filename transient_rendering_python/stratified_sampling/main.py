import numpy as np
import scipy.io
import sys, os
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time

from scipy.spatial import Delaunay

import pyigl as igl
import rendering

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  thread_num = 10
  thread_batch = 500
  normal = 'fn'
  method = 'n'
  lighting = np.array([[0.25, 0, 0]])
 
  measurement_num = lighting.shape[0] 
  sensor = lighting
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  vn = 0
  doublearea = igl.eigen.MatrixXd()

mesh = MESH()
#mesh_location = '../../mesh_processing/data/bunny.obj'
#read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)


[x, y] = np.meshgrid(np.linspace(-1, 1, 15), np.linspace(-1, 1, 15))
x = np.concatenate(x)
y = np.concatenate(y)
z = -0.5 + np.sqrt(4 - np.multiply(x+0.1,x+0.1) - np.multiply(y-0.2,y-0.2))
v = np.vstack((x, y, z)).T
tri = Delaunay(v[:,0:2])

mesh.vn = -np.vstack((x+0.1,y-0.2,z+0.5)).T/2
mesh.f = igl.eigen.MatrixXi(tri.simplices[:,[0,2,1]])
mesh.v = igl.eigen.MatrixXd(v)

igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
igl.doublearea(mesh.v, mesh.f, mesh.doublearea)

mesh.fn = np.array(mesh.fn)

#opt = OPT(100000)
opt = OPT(255)
opt.normal = 'n'
opt.method = 's'
measurement_num = 100
mesh_transient = np.zeros((opt.lighting.shape[0], opt.max_distance_bin))
tic = time.time()
mesh_transient = np.sum(rendering.render_all_collocate(mesh,opt,measurement_num),0)
mesh_transient /= measurement_num
print(time.time() - tic)

sample_num = opt.sample_num * measurement_num * opt.thread_batch

filename = os.getcwd() + '/sphere_mesh_sampling_s_3.mat'
scipy.io.savemat(filename, mdict={'lighting':opt.lighting, 'mesh_transient':mesh_transient, 'sample_num':sample_num})

