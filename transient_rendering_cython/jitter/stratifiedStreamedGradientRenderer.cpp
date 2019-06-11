/*
 * stratifiedStreamedTransientRenderer.cpp
 *
 *  Created on: Aug 28, 2018
 *      Author: igkiou
 */
#include "stratifiedStreamedGradientRenderer.h"

/* include embree API */
#include <embree3/rtcore.h>
#include <common/math/math.h>
#include <common/math/vec.h>
#include <common/math/affinespace.h>
//#include <math/linearspace2.h>
#include <common/core/ray.h>
#include <tasking/taskscheduler.h>
#include <algorithms/parallel_for.h>

#include "sampler.h"
#include "transient_and_gradient.h"
using namespace embree;

/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

void streamedTriangleCurvatureGrad(int triangleIndex, const float* vertices, const int* triangles, double* gradient) {
	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];
	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);
	
	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal) / 2;
	faceNormal /= 2* faceArea;
        
        Vec3fa e(v3-v2);
        Vec3fa g(cross(faceNormal,e/2));
        gradient[3*v1Ind] = g[0];
        gradient[3*v1Ind + 1] = g[1];
        gradient[3*v1Ind + 2] = g[2];

        e = v1 - v3;
        g = cross(faceNormal,e/2);
        gradient[3*v2Ind] = g[0];
        gradient[3*v2Ind + 1] = g[1];
        gradient[3*v2Ind + 2] = g[2];
     
        e = v2 - v1;
        g = cross(faceNormal,e/2);
        gradient[3*v3Ind] = g[0];
        gradient[3*v3Ind + 1] = g[1];
        gradient[3*v3Ind + 2] = g[2];
     
}
            
void streamedTriangleNormalArea(int triangleIndex, const float* vertices, const int* triangles, double* normal, double* area) {
	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];
	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);
	
	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal) / 2;
        area[triangleIndex] = faceArea;
	faceNormal /= 2* faceArea;
	normal[3 * triangleIndex] = faceNormal[0];
	normal[3 * triangleIndex+1] = faceNormal[1];
	normal[3 * triangleIndex+2] = faceNormal[2];

}

void streamedTriangleNormalSmoothing(int triangleIndex, const float* vertices, const int* triangles, double* normal, double* area, int* face_affinity, double* gradient, double* value) {
	Vec3fa n(normal[3*triangleIndex], normal[3*triangleIndex+1], normal[3*triangleIndex+2]);
        Vec3fa faceNormal(n);
	n *= area[triangleIndex];

	Vec3fa n1;
	for (int i = 0 ; i < 3; ++i) {
	  int f = face_affinity[3*triangleIndex + i];
	  if (f < 0)
	      continue;
	  n1[0] = normal[3*f];
	  n1[1] = normal[3*f+1];
	  n1[2] = normal[3*f+2];

	  n += n1*area[f];
	}
	n /= length(n);
        value[0] += area[triangleIndex] * (1-dot(n, faceNormal));

	faceNormal -= n;

	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];
	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);
	
        
        Vec3fa e(v3-v2);
        Vec3fa g(cross(faceNormal,e/2));
        gradient[3*v1Ind] = g[0];
        gradient[3*v1Ind + 1] = g[1];
        gradient[3*v1Ind + 2] = g[2];

        e = v1 - v3;
        g = cross(faceNormal,e/2);
        gradient[3*v2Ind] = g[0];
        gradient[3*v2Ind + 1] = g[1];
        gradient[3*v2Ind + 2] = g[2];
     
        e = v2 - v1;
        g = cross(faceNormal,e/2);
        gradient[3*v3Ind] = g[0];
        gradient[3*v3Ind + 1] = g[1];
        gradient[3*v3Ind + 2] = g[2];
     
}
double streamed_render_normal_smoothing(float* verticesD, int numVertices, int* trianglesD, int numTriangles, int* face_affinity, double* curvature_grad) {
    const int numThreads = (int) TaskScheduler::threadCount();
    double* normal = (double*) aligned_alloc(16, numTriangles*3*sizeof(double));
    double* area = (double*) aligned_alloc(16, numTriangles*sizeof(double));

    parallel_for( int(0), int(numTriangles), [&](const range<int>& range) {
        const int threadIndex = (int) TaskScheduler::threadIndex();
        for (int iterT=range.begin(); iterT < range.end(); ++iterT) {
            streamedTriangleNormalArea(iterT, verticesD, trianglesD, normal, area);
        }
    });

    double value = 0;
    double *values = (double*) aligned_alloc(16, numThreads*sizeof(double));
    double *grads = (double*) aligned_alloc(16, numThreads*numVertices*3*sizeof(double));
    std::memset(grads, 0, numThreads * numVertices * 3 *sizeof(grads));
    std::memset(curvature_grad, 0, numVertices*3*sizeof(curvature_grad));
    std::memset(values, 0, numThreads*sizeof(double));
    parallel_for( int(0), int(numTriangles), [&](const range<int>& range) {
        const int threadIndex = (int) TaskScheduler::threadIndex();
        for (int iterT=range.begin(); iterT < range.end(); ++iterT) {
            streamedTriangleNormalSmoothing(iterT, verticesD, trianglesD, normal, area, face_affinity, grads, &values[threadIndex]);
        }
    });

    for (int iterT = 0; iterT < numThreads; ++iterT) {
	value += values[iterT];
        for (int i = 0; i < 3 * numVertices; ++i) {
            curvature_grad[i] += grads[iterT*numVertices*3+i];
        }
    }
    free(grads);
    free(normal);
    free(area);
    return value;
}

void streamed_render_curvature_grad(float* verticesD, int numVertices, int* trianglesD, int numTriangles, double* curvature_grad) {
    
    const int numThreads = (int) TaskScheduler::threadCount();
    double *grads = (double*) aligned_alloc(16, numThreads*numVertices*3*sizeof(double));
    std::memset(grads, 0, numThreads * numVertices * 3 *sizeof(grads));
    std::memset(curvature_grad, 0, numVertices*3*sizeof(curvature_grad));
    parallel_for( int(0), int(numTriangles), [&](const range<int>& range) {
        const int threadIndex = (int) TaskScheduler::threadIndex();
        for (int iterT=range.begin(); iterT < range.end(); ++iterT) {
            streamedTriangleCurvatureGrad(iterT, verticesD, trianglesD, &grads[threadIndex*numVertices*3]);
        }
    });
    for (int iterT = 0; iterT < numThreads; ++iterT) {
        for (int i = 0; i < 3 * numVertices; ++i) {
            curvature_grad[i] += grads[iterT*numVertices*3+i];
        }
    }
    free(grads);
}


double streamed_render_gradient_albedo(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* albedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* jitter_weight, int weight_offset, int weight_length, double* transient, double *pathlengths, int refine_scale, int sigma_bin, int testing_flag, int loss_test) {
	/* start embree device */
	RTCDevice g_device = rtcNewDevice(nullptr);

	/* scene data */
	RTCScene g_scene = rtcNewScene(g_device);
	RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	/* set vertices */
	Vertex *vertices = (Vertex *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_VERTEX, 0,
								RTC_FORMAT_FLOAT3, sizeof(Vertex), numVertices);
	for (size_t iterVertex = 0; iterVertex < numVertices; ++iterVertex) {
		vertices[iterVertex] = {
			verticesD[iterVertex * 3],
			verticesD[iterVertex * 3 + 1],
			verticesD[iterVertex * 3 + 2]
		};
	}

	/* set triangles */
	Triangle *triangles = (Triangle *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_INDEX, 0,
								RTC_FORMAT_UINT3, sizeof(Triangle), numTriangles);
	for (size_t iterTriangle = 0; iterTriangle < numTriangles; ++iterTriangle) {
		triangles[iterTriangle] = {
			trianglesD[iterTriangle * 3],
			trianglesD[iterTriangle * 3 + 1],
			trianglesD[iterTriangle * 3 + 2]
		};
	}

	rtcCommitGeometry(mesh);
	rtcAttachGeometry(g_scene, mesh);
	rtcReleaseGeometry(mesh);

	/* set scene parameters */
	rtcSetSceneBuildQuality(g_scene, RTC_BUILD_QUALITY_HIGH);

	/* commit changes to scene */
	rtcCommitScene(g_scene);


	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

	for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
		pathlengths[iterBin] =
			(double) (pathlengthLowerBound + iterBin * pathlengthResolution);
	}
        int tmp_refine_scale = refine_scale;
	if (sigma_bin < 5) {
           tmp_refine_scale = 1;
        }        
        render_smoothed_transients(numBins, 
                                        measurement, 
                                        numTriangles, 
                                        g_device, 
                                        g_scene, 
                                        originD, 
                                        normalD, 
					verticesD,
					trianglesD,
					nullptr,
					albedo,
					pathlengthLowerBound,
					pathlengthUpperBound,
					pathlengthResolution,
					jitter_weight,
					weight_offset,
					weight_length,
					numSamples, 
                                        transient);
        double *difference = (double *) aligned_alloc(16, numBins * measurement * sizeof(double));
        
        std::transform(data, data+measurement*numBins, transient, difference, std::minus<double>());
        if (loss_test == 1) {
	  auto op = [](double& d) {d = 2*d*d*d;};
          std::for_each(difference, difference+measurement*numBins, op);
        }
        std::transform(difference, difference+measurement*numBins, weight, difference, std::multiplies<double>());
       
        double g = render_smoothed_gradients_albedo(numBins, 
                                  measurement, 
                                  refine_scale, 
                                  sigma_bin,  
                                  numTriangles, 
                                  numVertices,
                                  g_device, 
                                  g_scene, 
                                  originD, 
                                  normalD, 
				  verticesD,
				  trianglesD,
		                  nullptr,
				  albedo,
				  pathlengthLowerBound,
				  pathlengthUpperBound,
				  pathlengthResolution,
				  numSamples, 
                                  difference);
	

        free(difference);
	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);

        return g;
}

void streamed_render_gradient_w_albedo(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* albedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* jitter_weight, double* jitter_grad, int jitter_offset, int jitter_length, double* transient, double *pathlengths, double* gradient, int refine_scale, int sigma_bin, int testing_flag, int loss_test) {
	/* start embree device */
	RTCDevice g_device = rtcNewDevice(nullptr);

	/* scene data */
	RTCScene g_scene = rtcNewScene(g_device);
	RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	/* set vertices */
	Vertex *vertices = (Vertex *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_VERTEX, 0,
								RTC_FORMAT_FLOAT3, sizeof(Vertex), numVertices);
	for (size_t iterVertex = 0; iterVertex < numVertices; ++iterVertex) {
		vertices[iterVertex] = {
			verticesD[iterVertex * 3],
			verticesD[iterVertex * 3 + 1],
			verticesD[iterVertex * 3 + 2]
		};
	}

	/* set triangles */
	Triangle *triangles = (Triangle *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_INDEX, 0,
								RTC_FORMAT_UINT3, sizeof(Triangle), numTriangles);
	for (size_t iterTriangle = 0; iterTriangle < numTriangles; ++iterTriangle) {
		triangles[iterTriangle] = {
			trianglesD[iterTriangle * 3],
			trianglesD[iterTriangle * 3 + 1],
			trianglesD[iterTriangle * 3 + 2]
		};
	}

	rtcCommitGeometry(mesh);
	rtcAttachGeometry(g_scene, mesh);
	rtcReleaseGeometry(mesh);

	/* set scene parameters */
	rtcSetSceneBuildQuality(g_scene, RTC_BUILD_QUALITY_HIGH);

	/* commit changes to scene */
	rtcCommitScene(g_scene);


	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

	for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
		pathlengths[iterBin] =
			(double) (pathlengthLowerBound + iterBin * pathlengthResolution);
	}
        int tmp_refine_scale = refine_scale;
	if (sigma_bin < 5) {
           tmp_refine_scale = 1;
        }        
        render_smoothed_transients(numBins, 
                                        measurement, 
                                        numTriangles, 
                                        g_device, 
                                        g_scene, 
                                        originD, 
                                        normalD, 
					verticesD,
					trianglesD,
					nullptr,
					albedo,
					pathlengthLowerBound,
					pathlengthUpperBound,
					pathlengthResolution,
					jitter_weight,
					jitter_offset,
					jitter_length,
					numSamples, 
                                        transient);
        double *difference = (double *) aligned_alloc(16, numBins * measurement * sizeof(double));
        
        std::transform(data, data+measurement*numBins, transient, difference, std::minus<double>());
        if (loss_test == 1) {
	  auto op = [](double& d) {d = 2*d*d*d;};
          std::for_each(difference, difference+measurement*numBins, op);
        }
        std::transform(difference, difference+measurement*numBins, weight, difference, std::multiplies<double>());
       
        render_smoothed_gradients(numBins, 
                                  measurement, 
                                  numTriangles, 
                                  numVertices,
                                  g_device, 
                                  g_scene, 
                                  originD, 
                                  normalD, 
				  verticesD,
				  trianglesD,
		                  nullptr,
				  albedo,
				  pathlengthLowerBound,
				  pathlengthUpperBound,
				  pathlengthResolution,
				  jitter_weight,
				  jitter_grad,
			          jitter_offset,
				  jitter_length,
				  numSamples, 
                                  difference,
                                  gradient,
                                  testing_flag);
	

        free(difference);
	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}

void streamed_render_vertex_gradient(int vertex_num, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* gradient, int refine_scale, int sigma_bin) {
	/* start embree device */
	RTCDevice g_device = rtcNewDevice(nullptr);

	/* scene data */
	RTCScene g_scene = rtcNewScene(g_device);
	RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	/* set vertices */
	Vertex *vertices = (Vertex *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_VERTEX, 0,
								RTC_FORMAT_FLOAT3, sizeof(Vertex), numVertices);
	for (size_t iterVertex = 0; iterVertex < numVertices; ++iterVertex) {
		vertices[iterVertex] = {
			verticesD[iterVertex * 3],
			verticesD[iterVertex * 3 + 1],
			verticesD[iterVertex * 3 + 2]
		};
	}

	/* set triangles */
	Triangle *triangles = (Triangle *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_INDEX, 0,
								RTC_FORMAT_UINT3, sizeof(Triangle), numTriangles);
	for (size_t iterTriangle = 0; iterTriangle < numTriangles; ++iterTriangle) {
		triangles[iterTriangle] = {
			trianglesD[iterTriangle * 3],
			trianglesD[iterTriangle * 3 + 1],
			trianglesD[iterTriangle * 3 + 2]
		};
	}

	rtcCommitGeometry(mesh);
	rtcAttachGeometry(g_scene, mesh);
	rtcReleaseGeometry(mesh);

	/* set scene parameters */
	rtcSetSceneBuildQuality(g_scene, RTC_BUILD_QUALITY_HIGH);

	/* commit changes to scene */
	rtcCommitScene(g_scene);


	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

        render_smoothed_vertex_gradients(vertex_num,
				  numBins, 
                                  measurement, 
                                  refine_scale, 
                                  sigma_bin,  
                                  numTriangles, 
                                  numVertices,
                                  g_device, 
                                  g_scene, 
                                  originD, 
                                  normalD, 
				  verticesD,
				  trianglesD,
				  pathlengthLowerBound,
				  pathlengthUpperBound,
				  pathlengthResolution,
				  numSamples, 
                                  gradient);
	

	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}

void streamed_render_gradient(double* data, double* weight, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, float* vertexNormal, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* jitter_weight, double* jitter_grad, int jitter_offset, int jitter_length, double* transient, double *pathlengths, double* gradient, int testing_flag) {
	/* start embree device */
	RTCDevice g_device = rtcNewDevice(nullptr);

	/* scene data */
	RTCScene g_scene = rtcNewScene(g_device);
	RTCGeometry mesh = rtcNewGeometry(g_device, RTC_GEOMETRY_TYPE_TRIANGLE);

	/* set vertices */
	Vertex *vertices = (Vertex *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_VERTEX, 0,
								RTC_FORMAT_FLOAT3, sizeof(Vertex), numVertices);
	for (size_t iterVertex = 0; iterVertex < numVertices; ++iterVertex) {
		vertices[iterVertex] = {
			verticesD[iterVertex * 3],
			verticesD[iterVertex * 3 + 1],
			verticesD[iterVertex * 3 + 2]
		};
	}

	/* set triangles */
	Triangle *triangles = (Triangle *) rtcSetNewGeometryBuffer(
								mesh, RTC_BUFFER_TYPE_INDEX, 0,
								RTC_FORMAT_UINT3, sizeof(Triangle), numTriangles);
	for (size_t iterTriangle = 0; iterTriangle < numTriangles; ++iterTriangle) {
		triangles[iterTriangle] = {
			trianglesD[iterTriangle * 3],
			trianglesD[iterTriangle * 3 + 1],
			trianglesD[iterTriangle * 3 + 2]
		};
	}

	rtcCommitGeometry(mesh);
	rtcAttachGeometry(g_scene, mesh);
	rtcReleaseGeometry(mesh);

	/* set scene parameters */
	rtcSetSceneBuildQuality(g_scene, RTC_BUILD_QUALITY_HIGH);

	/* commit changes to scene */
	rtcCommitScene(g_scene);


	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

	for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
		pathlengths[iterBin] =
			(double) (pathlengthLowerBound + iterBin * pathlengthResolution);
	}
        render_smoothed_transients(numBins, 
                                        measurement, 
                                        numTriangles, 
                                        g_device, 
                                        g_scene, 
                                        originD, 
                                        normalD, 
					verticesD,
					trianglesD,
					vertexNormal,
					nullptr,
					pathlengthLowerBound,
					pathlengthUpperBound,
					pathlengthResolution,
					jitter_weight,
					jitter_offset,
					jitter_length,
					numSamples, 
                                        transient);
        double *difference = (double *) aligned_alloc(16, numBins * measurement * sizeof(double));
        
        std::transform(data, data+measurement*numBins, transient, difference, std::minus<double>());
        std::transform(difference, difference+measurement*numBins, weight, difference, std::multiplies<double>());
       
        render_smoothed_gradients(numBins, 
                                  measurement, 
                                  numTriangles, 
                                  numVertices,
                                  g_device, 
                                  g_scene, 
                                  originD, 
                                  normalD, 
				  verticesD,
				  trianglesD,
		                  vertexNormal,
				  nullptr,
				  pathlengthLowerBound,
				  pathlengthUpperBound,
				  pathlengthResolution,
			          jitter_weight,
			          jitter_grad,
				  jitter_offset,
                                  jitter_length,
				  numSamples, 
                                  difference,
                                  gradient,
                                  testing_flag);
	

        free(difference);
	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}
