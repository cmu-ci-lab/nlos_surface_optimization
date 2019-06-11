import numpy as np
import time
import sys, os
import math

import cgal_api

class MESH:
  v = 0
  f = 0


mesh = MESH()
z = .38 
v = np.array([[10, 10, 2], [11, 10, 2], [11, 11,2], [-.25, -.25, z], [.25, -.25, z], [.25, .25, z], [-.25, .25, z], [0, -.25, z], [.25, 0, z], [0, .25, z], [-.25,0,z], [0,0,.7]])
f = np.array([[0, 8,4], [0,7,8], [4,8,1],[1,8,5],[8,2,5],[8,6,2],[7,3,8],[3,6,8], [-3,-2,-1]]) + 3
#f = np.array([[0, 1, 2], [0,2,3]])
mesh.v = np.array(v, dtype=np.float32, order = 'C')
mesh.f = np.array(f, dtype=np.int32, order = 'C')       


v_num, f_num = cgal_api.keep_largest_connected_component(mesh.v, mesh.f, 0, 0)
print(v_num)
print(f_num)

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
