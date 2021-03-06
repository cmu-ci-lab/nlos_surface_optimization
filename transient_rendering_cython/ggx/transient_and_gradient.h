#ifndef TRANSIENT_AND_GRADIENT_H_
#define TRANSEINT_AND_GRADIENT_H_

#include <embree3/rtcore.h>
#include <common/core/ray.h>
#include "sampler.h"

using namespace embree;

double render_smoothed_gradients_alpha(int numBins, 
                                        int numSources, 
                                        int refine_scale,
                                        int sigma_bin,
                                        int numTriangles,
                                        int numVertices, 
                                        RTCDevice g_device, 
                                        RTCScene g_scene, 
                                        const float* originD, 
                                        const float* normalD, 
					const float *verticesD,
					const int32_t *trianglesD,
                                        const float* vertexNormal,
					const float *vertexAlbedo,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference);

void streamedRayTraceTriangle(int triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					const float *normals,
					const float *albedoes,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* transient, 
                                        int numTriangle);
void render_smoothed_transients( int numBins, 
                                        int numSources, 
                                        int refine_scale, 
                                        int sigma_bin,
                                        int numTriangles, 
                                        RTCDevice g_device, 
                                        RTCScene g_scene, 
                                        const float* originD, 
                                        const float* normalD, 
					const float *verticesD,
					const int32_t *trianglesD,
					const float *vertexNormal,
					const float *vertexAlbedo,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int numSamples, 
                                        double* transient);

void streamedRayTraceTriangleGradient(int32_t triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
                                        const float* normal,
					const float *albedoes,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference,
                                        double* gradient,
					int numTriangle,
                                        int numVertices,
                                        int refine_scale,
                                        int sigma_bin,
                                        double* weighting_kernal,
                                        double sigma, 
                                        int testing_flag);

void render_smoothed_gradients(int numBins, 
                                        int numSources, 
                                        int refine_scale, 
                                        int sigma_bin,
                                        int numTriangles,
                                        int numVertices, 
                                        RTCDevice g_device, 
                                        RTCScene g_scene, 
                                        const float* originD, 
                                        const float* normalD, 
					const float *verticesD,
					const int32_t *trianglesD,
                                        const float* normals,
					const float *vertexAlbedo,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference,
                                        double* gradient,
                                        int testing_flag);

void streamedRayTraceIntensity(int triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					const float *normals,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					int32_t numSamples, 
                                        double* intensity, 
                                        int numTriangle);

void render_intensity( int numSources, 
                                        int numTriangles, 
                                        RTCDevice g_device, 
                                        RTCScene g_scene, 
                                        const float* originD, 
                                        const float* normalD, 
					const float *verticesD,
					const int32_t *trianglesD,
					const float *vertexNormal,
					const float alpha,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					int numSamples, 
                                        double* intensity);
#endif  // transient_and_gradient.h
