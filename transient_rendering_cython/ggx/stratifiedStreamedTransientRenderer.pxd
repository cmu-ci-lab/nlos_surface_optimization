cdef extern from "stratifiedStreamedTransientRenderer.h":
  void streamed_render_intensity(float*, int, float*, float*, int, float*, int*, int, float, int, float, float, double*);
  void streamed_render_transient(float*, int, float*, float*, int, float*, float*, int*, int, float, int, float, float, float, double*, double *, int, int)
