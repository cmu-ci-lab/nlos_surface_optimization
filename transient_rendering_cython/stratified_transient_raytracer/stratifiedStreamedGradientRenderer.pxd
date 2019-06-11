cdef extern from "stratifiedStreamedGradientRenderer.h":
  void streamed_render_gradient(double*, float*, int, float*, float*, int, int*, int, int, float, float, float, int,  double*, double *, double*)
  void streamed_render_curvature_grad(float*, int, int*, int, double*)
