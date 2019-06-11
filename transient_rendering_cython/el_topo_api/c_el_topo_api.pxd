cdef extern from "c_el_topo_api.h":
  void el_topo_gradient(double*, int, int*, int, double*);
  void el_topo_remesh(double*, int&, int*, int&, double*, int, int*, int, double)
