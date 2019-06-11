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

#include "mkl_vsl.h"
#include "sampler.h"

using namespace embree;

int convolve(double* h, double* x, double* y, int nh, int nx, int ny, int iy0) {
    int status;
    VSLConvTaskPtr task;
    vsldConvNewTask1D(&task,VSL_CONV_MODE_AUTO,nh,nx,ny);
    vslConvSetStart(task, &iy0);
    status = vsldConvExec1D(task, h, 1, x, 1, y, 1);
    vslConvDeleteTask(&task);
    return status;
}

/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

void streamedRayTraceTriangleTransient(int triangle_source_index,
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
                                        double* transient, 
                                        int numTriangle) {
        Ray* rays = (Ray*) aligned_alloc(16, numSamples*sizeof(Ray));

        int triangleIndex = triangle_source_index % numTriangle;
        int sourceIndex = triangle_source_index / numTriangle;

	Vec3fa origin(originD[3*sourceIndex], originD[3*sourceIndex + 1], originD[3*sourceIndex + 2]);
	Vec3fa originNormal(originNormalD[3*sourceIndex], originNormalD[3*sourceIndex + 1], originNormalD[3*sourceIndex + 2]);

	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];
	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);

	/*
	 * TODO: Careful here, need to assume it faces towards the LOS wall.
	 * TODO: Perhaps replace with robust Orient2D formula from Shewchuk.
	 */
	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal) / 2;
	faceNormal /= 2* faceArea;

	Vec3fa n1(0, 0, 1), n2(0, 0, 1), n3(0, 0, 1);
	if (normals != nullptr) {
		n1 = Vec3fa(normals[3 * v1Ind], normals[3 * v1Ind + 1], normals[3 * v1Ind + 2]);
		n2 = Vec3fa(normals[3 * v2Ind], normals[3 * v2Ind + 1], normals[3 * v2Ind + 2]);
		n3 = Vec3fa(normals[3 * v3Ind], normals[3 * v3Ind + 1], normals[3 * v3Ind + 2]);
	}

	float a1(1), a2(1), a3(1);
	if (albedoes != nullptr) {
		a1 = albedoes[v1Ind];
		a2 = albedoes[v2Ind];
		a3 = albedoes[v3Ind];
	}

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

		/*
		 * TODO: Maybe replace with ld sequence.
		 * TODO: Is there a way to amortize the sampling cost here? E.g., use MKL?
		 * TODO: Maybe replace with C++ random?
		 */
		float S = sampler();
		float T = sampler();

		float sqrtT = embree::sqrt(T);
		float u = 1 - sqrtT;
		float v = (1 - S) * sqrtT;
		float w = S * sqrtT;
		Vec3fa point = u * v1 + v * v2 + w * v3;
		Vec3fa direction = normalize(point - origin);

		Ray& ray = rays[iterSample];
		init_Ray(ray, origin, direction, zero, inf);
	}

	/* intersect ray with scene */
	rtcIntersect1M(g_scene, &context, (RTCRayHit *) &rays[0], numSamples, sizeof(Ray));
	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
							/ pathlengthResolution);


	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

		if ((rays[iterSample].geomID != RTC_INVALID_GEOMETRY_ID) && (rays[iterSample].primID == triangleIndex)) {

			const float v = rays[iterSample].u;
			const float w = rays[iterSample].v;
			const float u = 1.0f - v - w;
			Vec3fa point = u * v1 + v * v2 + w * v3;
			float halfLength = length(point - origin);

			if ((halfLength <= pathlengthUpperBound / 2.0f)
						&& (halfLength >= pathlengthLowerBound / 2.0f)) {
				Vec3fa normal(faceNormal);
				if (normals != nullptr) {
					normal = u * n1 + v * n2 + w * n3;
				}
				float albedo(1);
				if (albedoes != nullptr) {
					albedo = u * a1 + v * a2 + w * a3;
				}
				float formFactor = - dot(normal, rays[iterSample].dir)
									* dot(originNormal, rays[iterSample].dir)
									/ halfLength
									/ halfLength;
				int32_t bin = (int32_t) embree::floor((2.0f * halfLength - pathlengthLowerBound)
														/ pathlengthResolution);
				transient[sourceIndex*numBins + bin] += (double) (faceArea * albedo * formFactor * formFactor)
								/ (double) numSamples;
			}
		}
	}
        free(rays);
}


void streamedRayTraceTriangleGradient_new(int32_t triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					const float *albedoes,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference,
                                        double* gradient,
					int numTriangle,
                                        int numVertices) {
	Ray *rays = (Ray *) aligned_alloc(16, numSamples  * sizeof(Ray));
        
	int triangleIndex = triangle_source_index % numTriangle;
        int sourceIndex = triangle_source_index / numTriangle;

	Vec3fa origin(originD[3*sourceIndex], originD[3*sourceIndex + 1], originD[3*sourceIndex + 2]);
	Vec3fa originNormal(originNormalD[3*sourceIndex], originNormalD[3*sourceIndex + 1], originNormalD[3*sourceIndex + 2]);

	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];
	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);

	/*
	 * TODO: Careful here, need to assume it faces towards the LOS wall.
	 * TODO: Perhaps replace with robust Orient2D formula from Shewchuk.
	 */
	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal)/2;
	faceNormal /= faceArea*2;
	float a1(1), a2(1), a3(1);
	if (albedoes != nullptr) {
		a1 = albedoes[v1Ind];
		a2 = albedoes[v2Ind];
		a3 = albedoes[v3Ind];
	}

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

		/*
		 * TODO: Maybe replace with ld sequence.
		 * TODO: Is there a way to amortize the sampling cost here? E.g., use MKL?
		 * TODO: Maybe replace with C++ random?
		 */
		float S = sampler();
		float T = sampler();

		float sqrtT = embree::sqrt(T);
		float u = 1 - sqrtT;
		float v = (1 - S) * sqrtT;
		float w = S * sqrtT;
		Vec3fa point = u * v1 + v * v2 + w * v3;
		Vec3fa direction = normalize(point - origin);

		Ray& ray = rays[iterSample];
		init_Ray(ray, origin, direction, zero, inf);
	}

	/* intersect ray with scene */
	rtcIntersect1M(g_scene, &context, (RTCRayHit *) &rays[0], numSamples, sizeof(Ray));
	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
							/ pathlengthResolution);


	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

		if ((rays[iterSample].geomID != RTC_INVALID_GEOMETRY_ID) && (rays[iterSample].primID == triangleIndex)) {
			const float v = rays[iterSample].u;
			const float w = rays[iterSample].v;
			const float u = 1.0f - v - w;
			Vec3fa point = u * v1 + v * v2 + w * v3;
			float halfLength = length(point - origin);

			if ((halfLength <= pathlengthUpperBound / 2.0f)
						&& (halfLength >= pathlengthLowerBound / 2.0f)) {
				Vec3fa normal(faceNormal);
				float albedo(1);
				if (albedoes != nullptr) {
					albedo = u * a1 + v * a2 + w * a3;
				}
                                float cos_theta2 = dot(originNormal, rays[iterSample].dir);
                                float cos_theta3 = dot(normal, -rays[iterSample].dir);
                                if (cos_theta2 < 0) cos_theta2 = 0;
                                if (cos_theta3 < 0) cos_theta3 = 0;
				float formFactor = cos_theta2 * cos_theta3/ halfLength
							     / halfLength;
				int32_t bin = (int32_t) embree::floor((2.0f * halfLength - pathlengthLowerBound)
														/ pathlengthResolution);
                                double intensity = albedo * formFactor *formFactor;

                                // compute gradient 
				Vec3fa t1 = 2 * cos_theta2 * cos_theta3 * (originNormal * cos_theta3 - normal * cos_theta2 + 4 * (-rays[iterSample].dir) * cos_theta2 * cos_theta3);
                                t1 /= embree::pow(halfLength,5);

                                Vec3fa t2 = normal*intensity;

                                Vec3fa gn = - 2 * rays[iterSample].dir * cos_theta3 * cos_theta2 * cos_theta2;
                                gn /= embree::pow(halfLength, 4);
                                float cos_tmp = dot(gn, normal);
                                gn -= normal*cos_tmp;
                                
                                t2 = (t2 + gn)/(2*faceArea);

                                Vec3fa e(v3-v2);
                                Vec3fa g(t1*u + cross(t2,e));
				g *= (-2)*difference[sourceIndex * numBins + bin];
                                gradient[3 * v1Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                gradient[3 * v1Ind + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                gradient[3 * v1Ind + 1] += (double) (faceArea * g[2])/ (double) numSamples;
                                e = v1-v3;
                                g = t1*v + cross(t2,e);
				g *= (-2)*difference[sourceIndex * numBins + bin];
                                gradient[3 * v2Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                gradient[3 * v2Ind + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                gradient[3 * v2Ind + 2] += (double) (faceArea * g[2])/ (double) numSamples;

                                e = v2-v1;
                                g = t1*w + cross(t2,e);
				g *= (-2)*difference[sourceIndex * numBins + bin];
                                gradient[3 * v3Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                gradient[3 * v3Ind + 2] += (double) (faceArea * g[1])/ (double) numSamples;
                                gradient[3 * v3Ind + 3] += (double) (faceArea * g[2])/ (double) numSamples;
			}
		}
	}
        free(rays);
}

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

void streamed_render_gradient(double* data, float* originD, int measurement, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, int width, double* transient, double *pathlengths, double* gradient) {
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

	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);

	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

	for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
		pathlengths[iterBin] =
			(double) (pathlengthLowerBound + iterBin * pathlengthResolution);
	}
	const int numThreads = (int) TaskScheduler::threadCount();
	smp::SamplerSet samplers(numThreads);

       
        double *difference = (double *) aligned_alloc(16, numBins * measurement * sizeof(double));
        std::memcpy(difference, data, numBins * measurement * sizeof(difference));
	
	double *transients = (double *) aligned_alloc(16, numThreads * numBins *sizeof(double) * measurement);
        if (transients == NULL) {
            printf("memory insufficient\n");
            return;
        }
	double *gradients = (double *) aligned_alloc(16, numThreads * 3 * numVertices * sizeof(gradients));
        if (gradients == NULL) {
            printf("grad memory insufficient\n");
            return;
        }

        std::memset(gradient, 0, 3 * numVertices * sizeof(gradient));
        std::memset(gradients, 0, numThreads * 3 * numVertices * sizeof(gradient));
	std::memset(transient, 0, numBins * measurement * sizeof(transient));
        std::memset(transients, 0, numThreads * numBins* sizeof(transient) * measurement);
	parallel_for(int(0), int(numTriangles*measurement),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangleTransient(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, nullptr, nullptr,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, &transients[threadIndex * measurement * numBins], numTriangles);
		}
	
	});


	parallel_for(int(0), int(measurement),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterSource = range.begin(); iterSource < range.end();															iterSource++) {
 
                	for(int iterThread = 0; iterThread < numThreads; ++iterThread) {
				for (int iterBin = 0; iterBin < numBins; ++ iterBin) {
					transient[iterSource*numBins + iterBin] += transients[iterThread * measurement * numBins + iterSource * numBins +  iterBin];
                                	difference[iterSource*numBins + iterBin] -= transients[iterThread * measurement * numBins + iterSource * numBins + iterBin];

				}
                    	}

                        if (width > 0) {

			    double *convolution_kernal = (double*) aligned_alloc(16, (2*width+1)*sizeof(double));
        		    double *y = (double*) aligned_alloc(16, (numBins+2*width)*sizeof(double));
        		    double *y2 = (double*) aligned_alloc(16, (numBins+2*width)*sizeof(double));
        		    std::fill(convolution_kernal, convolution_kernal + 2*width + 1, 1.0/((double) 2*width+1));
            		    std::memset(y, 0, (numBins+2*width)*sizeof(double));
           		    std::memset(y2, 0, (numBins+2*width)*sizeof(double));
            		    convolve(convolution_kernal, &difference[iterSource*numBins], y, 2*width+1, numBins, numBins+2*width, 0);
            		    convolve(convolution_kernal, &y[width], y2, 2*width+1, numBins, numBins+2*width, 0);
                        
			    std::memcpy(&difference[iterSource*numBins], &y2[width], numBins * sizeof(difference));
		 	    free(convolution_kernal);
                            free(y);
			    free(y2);
                        }
		}
	});

	parallel_for(int(0), int(numTriangles*measurement),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangleGradient_new(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, nullptr,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, difference, &gradients[threadIndex*3*numVertices], numTriangles, numVertices);
		}
	
	});
	for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		for (int32_t iterD = 0; iterD < numVertices * 3 ; ++iterD) {
	  		gradient[iterD] += gradients[iterThread * 3 * numVertices + iterD]/measurement;
		}
	}
        
        free(transients);
        free(gradients);
        free(difference);
	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}
