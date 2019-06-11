cdef extern from "stratifiedStreamedGradientRenderer.h":
  double streamed_render_gradient_alpha(double*, double*, float*, int, float*, float*, int, float*, int*, int, float, int, float, float, float, double*, double*, int, int);
  void streamed_render_gradient(double*, double*, float*, int, float*, float*, int, float*, int*, int, float, int, float, float, float, double*, double *, double*, int, int, int)
