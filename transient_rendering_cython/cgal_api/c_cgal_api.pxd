cdef extern from "c_cgal_api.h":
  void keep_largest_connected_component(float*, int&, int*, int&)
  void per_vertex_normal(float*, int, int*, int, float*)
  int find_convex_hull(float*, int, float*);
  void isotropic_remeshing(float*, int&, int*, int&, double, int, float*, int*, int, int)
  void border_vertex(float*, int, int*, int, int*)
  void face_affinity(float*, int, int*, int, int*)

