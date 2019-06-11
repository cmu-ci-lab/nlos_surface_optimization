cimport c_cgal_api

import cython
import numpy as np
cimport numpy as np


@cython.boundscheck(False)
@cython.wraparound(False)
def keep_largest_connected_component(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, int v_num, int f_num):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  
  c_cgal_api.keep_largest_connected_component(&v[0,0], v_num, &f[0,0], f_num)

  return v_num, f_num

@cython.boundscheck(False)
@cython.wraparound(False)
def per_vertex_normal(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[float, ndim=2, mode="c"] vn):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert vn.shape[0] == v_num, "vn should be of Vx3"
  assert vn.shape[1] == 3, "vn should be of Vx3"
  c_cgal_api.per_vertex_normal(&v[0,0], v_num, &f[0,0], f_num, &vn[0,0])



@cython.boundscheck(False)
@cython.wraparound(False)
def face_affinity(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[int, ndim=2, mode="c"] f_affinity):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert f_affinity.shape[0] == f_num, "f_affinity should be of Fx3"
  assert f_affinity.shape[1] == 3, "f_affinity should be of Fx3"
  c_cgal_api.face_affinity(&v[0,0], v_num, &f[0,0], f_num, &f_affinity[0,0])


@cython.boundscheck(False)
@cython.wraparound(False)
def border_vertex(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[int, ndim=1, mode="c"] v_idx):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert v_idx.shape[0] == v_num, "v_idx should be of size V"
  c_cgal_api.border_vertex(&v[0,0], v_num, &f[0,0], f_num, &v_idx[0]) 

@cython.boundscheck(False)
@cython.wraparound(False)
def isotropic_remeshing(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, double target_edge_length, int nb_iter, int v_num, int f_num, np.ndarray[float, ndim=2, mode="c"] new_v, np.ndarray[int, ndim=2, mode="c"] new_f):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert f.shape[1] == 3, "face should be Fx3"
  f_num = f.shape[0]
  assert new_v.shape[1] == 3, "new vertex should be V'x3"
  assert new_f.shape[1] == 3, "new face should be F'x3"
  
  c_cgal_api.isotropic_remeshing(&v[0,0], v_num, &f[0,0], f_num, target_edge_length, nb_iter, &new_v[0,0], &new_f[0,0], new_v.shape[0], new_f.shape[0])

  return v_num, f_num

@cython.boundscheck(False)
@cython.wraparound(False)
def find_convex_hull(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[float, ndim=2, mode="c"] result):
  assert v.shape[1] == 3, "vertex should be Vx3"
  v_num = v.shape[0]
  assert result.shape[1] == 2, "result should be Vx2"
  assert result.shape[0] == v_num, "result should be Vx2"
  return c_cgal_api.find_convex_hull(&v[0,0], v_num, &result[0,0])

