#ifndef STRATIFIED_STREAMED_GRAD_RENDERER
#define STRATIFIED_STREAMED_GRAD_RENDERER

void streamed_render_gradient(double* data, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, int w_width, double* transient, double *pathlengths, double* gradient);

void streamed_render_curvature_grad(float* verticesD, int numVertices, int* trianglesD, int numTriangles, double* curvature_grad);
#endif
