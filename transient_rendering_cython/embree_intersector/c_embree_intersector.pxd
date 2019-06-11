#import numpy as np
#cimport numpy as cnp

cdef extern from "c_embree_intersector.h":
  void barycentric_to_world(float*, int*, float*, int, float*);
  void embree3_tbb_line_intersection(float*, float*, int, float*, int, int*, int, float*)
  void embree3_tbb_short_line_intersection(float*, float*, int, float*, int, int*, int, float*)
