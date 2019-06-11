cimport c_el_topo_api

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def el_topo_remesh(np.ndarray[double, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[double, ndim=2, mode="c"] new_v, np.ndarray[int, ndim=2, mode='c'] new_f, int v_num, int f_num, double edge):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert new_v.shape[1] == 3, "new vertex should be V'x3"
  assert new_f.shape[1] == 3, "new face should be F'x3"
  c_el_topo_api.el_topo_remesh(&v[0,0], v_num, &f[0,0], f_num, &new_v[0,0], new_v.shape[0], &new_f[0,0], new_f.shape[0], edge)
  return v_num, f_num

@cython.boundscheck(False)
@cython.wraparound(False)
def el_topo_gradient(np.ndarray[double, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[double, ndim=2, mode="c"] new_v):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert new_v.shape[1] == 3, "new vertex should be Vx3"
  assert new_v.shape[0] == v_num, "new vertex should be Vx3"  
  c_el_topo_api.el_topo_gradient(&v[0,0], v_num, &f[0,0], f_num, &new_v[0,0])


