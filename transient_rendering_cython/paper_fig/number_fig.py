import numpy as np
import scipy.io
import sys, os


sys.path.insert(0,'/home/chiayint/research/libigl/python/')
sys.path.append('../embree_intersector/')

import math
import time
import argparse
import torch
import pyigl as igl

import embree_intersector


class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  f_affinity = 0
  v_edge = 0
  alpha = 0.1

folder_name = os.getcwd() + '/progress-b-3/'
if not os.path.isdir(folder_name):
  os.mkdir(folder_name)



mesh = MESH()
mesh_init_location = os.getcwd() + '/init/cnlos_numbers3_threshold.obj'

igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       


[x, y] = np.meshgrid(np.linspace(-.4, .4, 1024), np.linspace(0.01, 0.01, 1))
x = np.concatenate(x)
y = np.concatenate(y)
o = np.vstack((x, y, 0*np.ones_like(x))).T
o = np.array(o, dtype=np.float32, order='C')

d = np.array([0,0,1])
d = np.tile(d, (o.shape[0],1))
d = np.array(d, dtype=np.float32, order = 'C')

b1 = np.ndarray((o.shape[0],3), dtype=np.float32)

embree_intersector.embree3_tbb_intersection(o,d,mesh.v,mesh.f,b1)
p1 = np.ndarray((o.shape[0],3), dtype=np.float32)
embree_intersector.barycoord_to_world(mesh.v,mesh.f,b1,p1)

mesh = MESH()
mesh_init_location = os.getcwd() + '/init/final_number.obj'

igl.readOBJ(mesh_init_location, mesh.v, mesh.f)
mesh.v = np.array(mesh.v, dtype=np.float32, order = 'C')
mesh.f = np.array(mesh.f, dtype=np.int32, order = 'C')       

b2 = np.ndarray((o.shape[0],3), dtype=np.float32)

embree_intersector.embree3_tbb_intersection(o,d,mesh.v,mesh.f,b2)
p2 = np.ndarray((o.shape[0],3), dtype=np.float32)
embree_intersector.barycoord_to_world(mesh.v,mesh.f,b2,p2)

filename = folder_name + 'data.mat'
scipy.io.savemat(filename, mdict={'p1': p1,'p2':p2, 'b1': b1, 'b2': b2})

