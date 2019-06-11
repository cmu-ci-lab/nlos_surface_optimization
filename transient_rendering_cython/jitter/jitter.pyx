cimport stratifiedStreamedTransientRenderer
cimport stratifiedStreamedGradientRenderer
import math

import cython
import numpy as np
cimport numpy as np




'''
@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedGradientAlbedo(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[int, ndim=2, mode='c'] faces, np.ndarray[float, ndim=1, mode='c'] albedo, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode='c'] data, np.ndarray[double,ndim=2,mode='c'] weight, int refine_scale, int sigma_bin, int testing_flag, int loss_flag): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert albedo.shape[0] == vertices.shape[0], "albedo needs to be Vx1"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert data.shape[0] == L, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert data.shape[1] == numBins, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert weight.shape[0] == L, "weighting should be LxB"
  assert weight.shape[1] == numBins, "weighting should be LxB" 
  return stratifiedStreamedGradientRenderer.streamed_render_gradient_albedo(&data[0,0], &weight[0,0],  &origin[0, 0], L, &normal[0,0], &vertices[0,0], vertices.shape[0], &albedo[0], &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution,  &transient[0,0], &pathlengths[0], refine_scale, sigma_bin, testing_flag, loss_flag)

@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedGradientWithAlbedo(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[int, ndim=2, mode='c'] faces, np.ndarray[float, ndim=1, mode='c'] albedo, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode = 'c'] gradient, np.ndarray[double, ndim=2, mode='c'] data, np.ndarray[double,ndim=2,mode='c'] weight, int refine_scale, int sigma_bin, int testing_flag, int loss_flag): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert albedo.shape[0] == vertices.shape[0], "albedo needs to be Vx1"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert gradient.shape[0] == vertices.shape[0], "gradient dimension should be Vx3"
  assert gradient.shape[1] == 3, "gradient dimension should be Vx3"
  assert data.shape[0] == L, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert data.shape[1] == numBins, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert weight.shape[0] == L, "weighting should be LxB"
  assert weight.shape[1] == numBins, "weighting should be LxB" 
  stratifiedStreamedGradientRenderer.streamed_render_gradient_w_albedo(&data[0,0], &weight[0,0],  &origin[0, 0], L, &normal[0,0], &vertices[0,0], vertices.shape[0], &albedo[0], &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution,  &transient[0,0], &pathlengths[0], &gradient[0,0], refine_scale, sigma_bin, testing_flag, loss_flag)

'''

@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedGradient(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[int, ndim=2, mode='c'] faces, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] jitter_weight, np.ndarray[double, ndim=2, mode='c'] jitter_grad, int jitter_offset, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode = 'c'] gradient, np.ndarray[double, ndim=2, mode='c'] data, np.ndarray[double,ndim=2,mode='c'] weight, int testing_flag): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert gradient.shape[0] == vertices.shape[0], "gradient dimension should be Vx3"
  assert gradient.shape[1] == 3, "gradient dimension should be Vx3"
  assert data.shape[0] == L, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert data.shape[1] == numBins, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert weight.shape[0] == L, "weighting should be LxB"
  assert weight.shape[1] == numBins, "weighting should be LxB" 
  stratifiedStreamedGradientRenderer.streamed_render_gradient(&data[0,0], &weight[0,0],  &origin[0, 0], L, &normal[0,0], &vertices[0,0], vertices.shape[0], NULL, &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution, &jitter_weight[0,0], &jitter_grad[0,0], jitter_offset, jitter_weight.shape[0], &transient[0,0], &pathlengths[0], &gradient[0,0], testing_flag)

'''
@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedShadingGradient(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[int, ndim=2, mode='c'] faces, np.ndarray[float, ndim=2, mode='c'] vertexNormal, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode = 'c'] gradient, np.ndarray[double, ndim=2, mode='c'] data, np.ndarray[double,ndim=2,mode='c'] weight, int refine_scale, int sigma_bin, int testing_flag, int loss_flag): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert vertexNormal.shape[1] == 3, "vertex normal needs to be Vx3"
  assert vertices.shape[0] == vertexNormal.shape[0], "vertex normal needs to be Vx3"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert gradient.shape[0] == vertices.shape[0], "gradient dimension should be Vx3"
  assert gradient.shape[1] == 3, "gradient dimension should be Vx3"
  assert data.shape[0] == L, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert data.shape[1] == numBins, "data transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert weight.shape[0] == L, "weighting should be LxB"
  assert weight.shape[1] == numBins, "weighting should be LxB" 
  stratifiedStreamedGradientRenderer.streamed_render_gradient(&data[0,0], &weight[0,0],  &origin[0, 0], L, &normal[0,0], &vertices[0,0], vertices.shape[0], &vertexNormal[0,0], &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution,  &transient[0,0], &pathlengths[0], &gradient[0,0], refine_scale, sigma_bin, testing_flag, loss_flag)
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedTransientShading(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[float, ndim=2, mode='c'] vertexNormal, np.ndarray[int, ndim=2, mode='c'] faces, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode='c'] weight, int weight_offset): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert vertexNormal.shape[1] == 3, "vertex normal needs to be Vx3"
  assert vertices.shape[0] == vertexNormal.shape[0], "vertex normal needs to be Vx3"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
 
  stratifiedStreamedTransientRenderer.streamed_render_transient(&origin[0, 0], L, &normal[0, 0], &vertices[0,0], vertices.shape[0], &vertexNormal[0,0], NULL, &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution, &weight[0,0], weight_offset, weight.shape[0], &transient[0, 0], &pathlengths[0])
  
@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedTransientwAlbedo(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[float, ndim=1, mode='c'] albedo, np.ndarray[int, ndim=2, mode='c'] faces, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode='c'] weight, int weight_offset): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert vertices.shape[0] == albedo.shape[0], "albedo nees to be Vx1"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
 
  stratifiedStreamedTransientRenderer.streamed_render_transient(&origin[0, 0], L, &normal[0, 0], &vertices[0,0], vertices.shape[0], NULL, &albedo[0], &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution, &weight[0,0], weight_offset, weight.shape[0], &transient[0, 0], &pathlengths[0])
  

@cython.boundscheck(False)
@cython.wraparound(False)
def renderStreamedTransient(np.ndarray[float, ndim=2, mode='c'] origin, np.ndarray[float, ndim=2, mode='c'] normal, np.ndarray[float, ndim=2, mode='c'] vertices, np.ndarray[int, ndim=2, mode='c'] faces, int num_sample, float lower_bound, float upper_bound, float resolution, np.ndarray[double, ndim=2, mode='c'] transient, np.ndarray[double, ndim=1, mode='c'] pathlengths, np.ndarray[double, ndim=2, mode='c'] weight, int weight_offset): 
  L = origin.shape[0]
  assert origin.shape[1] == 3, "origin needs to be Lx3"
  assert normal.shape[0] == L, "normal needs to be Lx3"
  assert normal.shape[1] == 3, "normal needs to be Lx3"
  assert vertices.shape[1] == 3, "vertices needs to be Vx3"
  assert faces.shape[1] == 3, "faces needs to be Fx3"
  numBins = math.ceil((upper_bound - lower_bound)/resolution)
  assert transient.shape[0] == L, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert transient.shape[1] == numBins, "transient dimension should  be LxB   (B = math.ceil((upper_bound-lower_bound)/resolution))"
  assert pathlengths.shape[0] == numBins, "pathlength dimension should be Bx1 (B = math.ceil((upper_bound-lower_bound)/resolution))"
 
  stratifiedStreamedTransientRenderer.streamed_render_transient(&origin[0, 0], L, &normal[0, 0], &vertices[0,0], vertices.shape[0], NULL, NULL, &faces[0,0], faces.shape[0], num_sample, lower_bound, upper_bound, resolution, &weight[0,0], weight_offset, weight.shape[0], &transient[0, 0], &pathlengths[0])
  
