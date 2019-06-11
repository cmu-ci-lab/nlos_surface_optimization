import numpy as np
import time

import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')

import pyigl as igl

import embree_intersector

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

mesh_location = '../../mesh_processing/data/bunny.obj'
mesh = MESH()
read_file = igl.readOBJ(mesh_location, mesh.v, mesh.f)
v = np.array(mesh.v, dtype = np.float32, order='C')
f = np.array(mesh.f, dtype = np.int32, order = 'C')

test_mesh = embree_intersector.PyMesh(v,f)

resolution = 625*8
#resolution = 10
[x, y] = np.meshgrid(np.linspace(-0.5, 0.5, resolution), np.linspace(-0.5, 0.5, resolution))
x = np.concatenate(x)
y = np.concatenate(y)
o = np.vstack((x, y, -1*np.ones_like(x))).T
o_igl = igl.eigen.MatrixXd(o)
o = np.array(o, dtype=np.float32, order='C')

d = np.array([0,0,1])
d = np.tile(d, (o.shape[0],1))
d_igl = igl.eigen.MatrixXd(d.astype(float))

d = np.array(d, dtype=np.float32, order = 'C')

triid = np.ndarray(o.shape[0], dtype=np.float32)
tic = time.time()
embree_intersector.embree3_tbb_short_intersection(o[range(1),:],d[range(1),:],v,f,triid[range(1)])
print('dummy tbb 1 dim %f' %(time.time() - tic))

#tic = time.time()
#barycoord_igl = igl.embree.line_mesh_intersection(o_igl, d_igl, mesh.v, mesh.f)
#print('igl 3 dim %f' % (time.time() - tic))


barycoord_tbb_full = np.ndarray((o.shape[0],3), dtype=np.float32)
tic = time.time()
embree_intersector.embree3_tbb_intersection(o,d,v,f,barycoord_tbb_full)
print('embree3_tbb 3 dim %f' %(time.time() - tic))

barycoord_tbb_mesh_full = np.ndarray((o.shape[0],3), dtype=np.float32)
tic = time.time()
test_mesh.embree3_tbb_intersection(o,d,barycoord_tbb_mesh_full)
print('embree3_tbb_mesh 3 dim %f' %(time.time() - tic))



intersection_p = np.ndarray((o.shape[0],3), dtype=np.float32)
tic = time.time()
embree_intersector.barycoord_to_world(v,f,barycoord_tbb_full,intersection_p)
print('barycoord to world %f'%(time.time() - tic))

intersection_p = np.ndarray((o.shape[0],3), dtype=np.float32)
tic = time.time()
test_mesh.barycoord_to_world(barycoord_tbb_mesh_full,intersection_p)
print('barycoord to world mesh %f'%(time.time() - tic))




barycoord_tbb = np.ndarray(o.shape[0], dtype=np.float32)
tic = time.time()
embree_intersector.embree3_tbb_short_intersection(o,d,v,f,barycoord_tbb)
print('embree3_tbb 1 dim %f' %(time.time() - tic))

barycoord_tbb_mesh = np.ndarray(o.shape[0], dtype=np.float32)
tic = time.time()
test_mesh.embree3_tbb_short_intersection(o,d,barycoord_tbb_mesh)
print('embree3_tbb_mesh 1 dim %f' %(time.time() - tic))

print ('diff%f'%np.linalg.norm(barycoord_tbb-barycoord_tbb_mesh))

#barycoord_full = np.ndarray((o.shape[0],3), dtype=np.float32)
#tic = time.time()
#p_test_embree.embree3_intersection(o,d,v,f,barycoord_full)
#print('embree3 3 dim %f' %(time.time() - tic))


#barycoord = np.ndarray((o.shape[0],1), dtype=np.float32)
#tic = time.time()
#p_test_embree.embree3_intersection(o,d,v,f,barycoord)
#print('embree3 1 dim %f' %(time.time() - tic))


