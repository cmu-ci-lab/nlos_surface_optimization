cdef extern from "stratifiedStreamedGradientRenderer.h":
#  void streamed_render_gradient_w_albedo(double*, double*, float*, int, float*, float*, int, float*, int*, int, int, float, float, float, double*, double *, double*, int, int, int, int)
#  double streamed_render_gradient_albedo(double*, double*, float*, int, float*, float*, int, float*, int*, int, int, float, float, float, double*, double *, int, int, int, int)
  void streamed_render_gradient(double*, double*, float*, int, float*, float*, int, float*, int*, int, int, float, float, float, double*, double*, int, int, double*, double *, double*, int)
#  void streamed_render_vertex_gradient(int, float*, int, float*, float*, int, int*, int, int, float, float, float, double*, int, int)
#  void streamed_render_curvature_grad(float*, int, int*, int, double*)
#  double streamed_render_normal_smoothing(float*, int, int*, int, int*, double*)
