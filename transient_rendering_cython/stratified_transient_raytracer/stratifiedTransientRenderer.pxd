cdef extern from "stratifiedTransientRenderer.h":
  void render_transient(float*, float*, float*, int, int*, int, int, float, float, float, double*, double *)
