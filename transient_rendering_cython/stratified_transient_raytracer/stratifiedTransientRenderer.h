#ifndef STRATIFIED_TRANSIENT_RENDERER
#define STRATIFIED_TRANSIENT_RENDERER

/*
#include "sampler.h"
#include <embree3/rtcore.h>
void rayTraceTriangle(int32_t triangleIndex,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					const float *normals,
					const float *albedoes,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples,
					double *transient);
*/
void render_transient(float* originD, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths);

#endif
