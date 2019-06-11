cdef extern from "stratifiedStreamedTransientRenderer.h":
  void streamed_render_transient(float*, int, float*, float*, int, float*, float*, int*, int, int, float, float, float, double*, int, int, double*, double *)
