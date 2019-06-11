#ifndef STRATIFIED_STREAMED_GRAD_RENDERER
#define STRATIFIED_STREAMED_GRAD_RENDERER
double streamed_render_gradient_albedo(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* albedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths, int refine_scale, int sigma_bin, int testing_flag, int loss_test);

void streamed_render_gradient_w_albedo(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* albedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths, double* gradient, int refine_scale, int sigma_bin, int testing_flag, int loss_test);

void streamed_render_gradient(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* vertexNormal, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths, double* gradient, int refine_scale, int sigma_bin, int testing_flag, int loss_test);

void streamed_render_vertex_gradient(int vertex_num, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* gradient, int refine_scale, int sigma_bin);

double streamed_render_normal_smoothing(float* verticesD, int numVertices, int* trianglesD, int numTriangles, int* face_affinity, double* curvature_grad);

void streamed_render_curvature_grad(float* verticesD, int numVertices, int* trianglesD, int numTriangles, double* curvature_grad);
#endif
