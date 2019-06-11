#ifndef STRATIFIED_STREAMED_TRANSIENT_RENDERER
#define STRATIFIED_STREAMED_TRANSIENT_RENDERER
void streamed_render_transient(float* originD, int numSources, float* normalD, float* verticesD, int numVertices, float* vertexNormal, float* vertexAlbedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* weight, int weight_offset, int weight_length, double* transient, double *pathlengths);
#endif
