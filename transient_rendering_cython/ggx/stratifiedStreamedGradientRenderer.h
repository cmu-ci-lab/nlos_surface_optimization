#ifndef STRATIFIED_STREAMED_GRAD_RENDERER
#define STRATIFIED_STREAMED_GRAD_RENDERER

double streamed_render_gradient_alpha(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* vertexNormal, int* trianglesD, int numTriangles, float alpha, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths, int refine_scale, int sigma_bin);

void streamed_render_gradient(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* vertexNormal, int* trianglesD, int numTriangles, float alpha,  int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths, double* gradient, int refine_scale, int sigma_bin, int testing_flag);
#endif
