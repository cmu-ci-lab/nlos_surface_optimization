import numpy as np
import scipy.io
import time
import sys, os
import math

import el_topo_api

class MESH:
  v = 0
  f = 0


mesh = MESH()
'''
z = .38 
v = np.array([[-.25, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [0, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,.7]])
#v = np.array([[10, 10, 2], [11, 10, 2], [11, 11,2], [-.25, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [0, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,.7]])
f = np.array([[0, 8,4], [0,7,8], [4,8,1],[1,8,5],[8,2,5],[8,6,2],[7,3,8],[3,6,8]]) 
#f = np.array([[0, 1, 2], [0,2,3]])
'''
v = np.array([[1,0,0], [0,0,0], [0,0,1], [1,0,1], [0,1,0], [0,1,1], [0.1, .5, .4]])
new_v = np.array([[1,0,0], [0,0,0], [0,0,1], [1,0,1], [0,1,0], [0,1,1], [-.2, .5, .4]])
f = np.array([[0,1,2], [0,2,3], [1,4,5], [1,5,2], [0,6,3]])
mesh.v = np.array(v, dtype=np.double, order = 'C')
mesh.f = np.array(f, dtype=np.int32, order = 'C')       


new_v = np.array(new_v, dtype=np.double, order = 'C')
el_topo_api.el_topo_gradient(mesh.v, mesh.f, new_v)
mesh.v = new_v
print(mesh.v)



new_v = np.empty((100,3), dtype = np.double, order='C')
new_f = np.empty((100,3), dtype = np.int32, order='C')

v_num, f_num = el_topo_api.el_topo_remesh(mesh.v, mesh.f, new_v, new_f, 0, 0, 0.6)
print(new_v[range(v_num),:])
print(new_f[range(f_num),:])

#filename = os.getcwd() + '/remesh.mat'
#scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'new_v': new_v[range(v_num),:], 'new_f': new_f[range(f_num),:]})

'''

z = .5
new_v = np.array([[-.1, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [-.2, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,.7]])
new_v = np.array(new_v, dtype=np.double, order = 'C')
el_topo_api.el_topo_gradient(mesh.v, mesh.f, new_v)
print(new_v)
'''


#vn = np.empty(mesh.v.shape, dtype=np.float32, order = 'C')
#cgal_api.per_vertex_normal(mesh.v, mesh.f, vn)
#print(vn)

#f_idx = -1* np.ones(f.shape, dtype=np.int32, order='C')
#cgal_api.face_affinity(mesh.v, mesh.f, f_idx)
#print(f_idx)

#v_idx = np.zeros(v.shape[0], dtype=np.int32, order='C')
#cgal_api.border_vertex(mesh.v, mesh.f, v_idx)
#print(v_idx)
'''

target_edge_length = 0.2
nb_iter = 3

v = np.empty((100,3), dtype=np.float32, order = 'C')
f = np.empty((100,3), dtype=np.int32, order = 'C')

num1, num2= cgal_api.isotropic_remeshing(mesh.v, mesh.f, target_edge_length, nb_iter, 0, 0, v, f)

v = v[range(num1),:]
f = f[range(num2),:]
'''
'''
print(num1)
print(v[range(num1),:])
result = np.empty((v.shape[0],2), dtype=np.float32, order='C')

result_num = cgal_api.find_convex_hull(mesh.v, result)

print(result_num)
print(result[range(result_num),:])

'''
