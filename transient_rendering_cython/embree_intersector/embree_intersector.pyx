cimport c_embree_intersector
from c_mesh cimport Mesh

import cython
import numpy as np
cimport numpy as np

cdef class PyMesh:
  cdef Mesh c_mesh
  def __cinit__(self, np.ndarray[float, ndim=2, mode='c'] v, np.ndarray[int, ndim=2, mode='c'] f):
    assert v.shape[1] == 3, "Vertices needs to be Vx3"
    assert f.shape[1] == 3, "Face needs to be Tx3"
    self.c_mesh = Mesh(&v[0,0], v.shape[0], &f[0,0], f.shape[0])
  def test(self):
    self.c_mesh.test()
  
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def embree3_tbb_intersection(self, np.ndarray[float, ndim=2, mode="c"] origin, np.ndarray[float, ndim=2, mode="c"] direction, np.ndarray[float, ndim=2, mode="c"] barycoord):
    assert origin.shape[0] == direction.shape[0], "Origin and Direction need to be Nx3"
    assert origin.shape[1] == 3, "Origin needs to be Nx3"
    assert direction.shape[1] == 3, "Direction needs to be Nx3"
    assert barycoord.shape[0] == origin.shape[0], "barycoord needs to be Nx1 or Nx3"
    assert barycoord.shape[1] == 3, "barycoord needs to be Nx1 or Nx3"  
    self.c_mesh.embree3_tbb_line_intersection(&origin[0,0], &direction[0,0], direction.shape[0], &barycoord[0,0])
  
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def embree3_tbb_short_intersection(self, np.ndarray[float, ndim=2, mode="c"] origin, np.ndarray[float, ndim=2, mode="c"] direction, np.ndarray[float, ndim=1, mode="c"] barycoord):
    assert origin.shape[0] == direction.shape[0], "Origin and Direction need to be Nx3"
    assert origin.shape[1] == 3, "Origin needs to be Nx3"
    assert direction.shape[1] == 3, "Direction needs to be Nx3"
    assert barycoord.shape[0] == origin.shape[0], "barycoord needs to be Nx1"
    self.c_mesh.embree3_tbb_short_line_intersection(&origin[0,0], &direction[0,0], direction.shape[0], &barycoord[0])
  
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def set_vn(self, np.ndarray[float, ndim=2, mode="c"] vn):
    assert vn.shape[1] == 3, "vn needs to be #vertices x 3"
    assert vn.shape[0] == self.c_mesh.get_vertex_num(), "vn nees to be #vertices x 3"
    self.c_mesh.set_vn(&vn[0,0])

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def set_fn_and_face_area(self, np.ndarray[float, ndim=2, mode="c"] fn, np.ndarray[float, ndim=1, mode="c"] area):
    assert fn.shape[1] == 3, "fn needs to be #face x 3"
    face_num = self.c_mesh.get_face_num()
    assert fn.shape[0] == face_num, "fn needs to be #face x 3"
    assert area.shape[0] == face_num, "barycoord needs to be #face x 1"
    self.c_mesh.set_fn_and_face_area(&fn[0,0], &area[0])
  
 
  @cython.boundscheck(False)
  @cython.wraparound(False)
  def barycoord_to_world(self, np.ndarray[float, ndim=2, mode="c"] barycoord, np.ndarray[float, ndim=2, mode="c"] intersection_p):  
    assert barycoord.shape[0] == intersection_p.shape[0], "barycoord and intersection_p should be Nx3"
    assert barycoord.shape[1] == 3, "barycoord should be Nx3"
    assert intersection_p.shape[1] == 3, "intersection_p should be Nx3"
    self.c_mesh.barycentric_to_world(&barycoord[0,0], barycoord.shape[0], &intersection_p[0,0]) 


@cython.boundscheck(False)
@cython.wraparound(False)
def barycoord_to_world(np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[float, ndim=2, mode="c"] barycoord, np.ndarray[float, ndim=2, mode="c"] intersection_p):  
  assert v.shape[1] == 3, "vertex should be Vx3"
  assert f.shape[1] == 3, "face should be Fx3"
  assert barycoord.shape[0] == intersection_p.shape[0], "barycoord and intersection_p should be Nx3"
  assert barycoord.shape[1] == 3, "barycoord should be Nx3"
  assert intersection_p.shape[1] == 3, "intersection_p should be Nx3"
  c_embree_intersector.barycentric_to_world(&v[0,0], &f[0,0], &barycoord[0,0], barycoord.shape[0], &intersection_p[0,0]) 

@cython.boundscheck(False)
@cython.wraparound(False)
def embree3_tbb_short_intersection(np.ndarray[float, ndim=2, mode="c"] origin, np.ndarray[float, ndim=2, mode="c"] direction, np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[float, ndim=1, mode="c"] barycoord):
  assert v.shape[1] == 3, "vertex should be Vx3"
  assert f.shape[1] == 3, "face should be Fx3"
  assert origin.shape[0] == direction.shape[0], "Origin and Direction need to be Nx3"
  assert origin.shape[1] == 3, "Origin needs to be Nx3"
  assert direction.shape[1] == 3, "Direction needs to be Nx3"
  assert barycoord.shape[0] == origin.shape[0], "barycoord needs to be Nx1"
  c_embree_intersector.embree3_tbb_short_line_intersection(&origin[0,0], &direction[0,0], direction.shape[0], &v[0,0], v.shape[0], &f[0,0], f.shape[0], &barycoord[0])
  
@cython.boundscheck(False)
@cython.wraparound(False)
def embree3_tbb_intersection(np.ndarray[float, ndim=2, mode="c"] origin, np.ndarray[float, ndim=2, mode="c"] direction, np.ndarray[float, ndim=2, mode="c"] v, np.ndarray[int, ndim=2, mode="c"] f, np.ndarray[float, ndim=2, mode="c"] barycoord):

  assert v.shape[1] == 3, "vertex should be Vx3"
  assert f.shape[1] == 3, "face should be Fx3"
  assert origin.shape[0] == direction.shape[0], "Origin and Direction need to be Nx3"
  assert origin.shape[1] == 3, "Origin needs to be Nx3"
  assert direction.shape[1] == 3, "Direction needs to be Nx3"
  assert barycoord.shape[0] == origin.shape[0], "barycoord needs to be Nx1 or Nx3"
  assert barycoord.shape[1] == 3, "barycoord needs to be Nx3"  
  c_embree_intersector.embree3_tbb_line_intersection(&origin[0,0], &direction[0,0], direction.shape[0], &v[0,0], v.shape[0], &f[0,0], f.shape[0], &barycoord[0,0])
  
