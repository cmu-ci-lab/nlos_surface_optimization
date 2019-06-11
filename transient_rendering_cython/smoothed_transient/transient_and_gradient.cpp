#include "transient_and_gradient.h"


#include <embree3/rtcore.h>
#include <common/math/math.h>
#include <common/math/vec.h>
#include <common/math/affinespace.h>
#include <common/core/ray.h>

#include <tasking/taskscheduler.h>
#include <algorithms/parallel_for.h>

#include "convolution_mkl.h"
#include "constants.h"
#include "sampler.h"

using namespace embree;
/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

void streamedRayTraceIntensity(int triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					const float *normals,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					int32_t numSamples, 
                                        double* intensity, 
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

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

		/*
		 * TODO: Maybe replace with ld sequence.
		 * TODO: Is there a way to amortize the sampling cost here? E.g., use MKL?
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
				float formFactor = - dot(normal, rays[iterSample].dir)
									* dot(originNormal, rays[iterSample].dir)
									/ halfLength
									/ halfLength;
				formFactor = embree::max((float)0.0, formFactor);
                                intensity[triangleIndex] += (double) (faceArea * albedo * formFactor * formFactor)
								/ (double) numSamples;
			}
		}
	}
        free(rays);
}


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
				formFactor = embree::max((float) 0.0, formFactor);
				int32_t bin = (int32_t) embree::floor((2.0f * halfLength - pathlengthLowerBound)
														/ pathlengthResolution);
				transient[sourceIndex*numBins + bin] += (double) (faceArea * albedo * formFactor * formFactor)
								/ (double) numSamples;
			}
		}
	}
        free(rays);
}

void render_intensity( int numSources, 
                                        int numTriangles, 
                                        RTCDevice g_device, 
                                        RTCScene g_scene, 
                                        const float* originD, 
                                        const float* normalD, 
					const float *verticesD,
					const int32_t *trianglesD,
					const float *vertexNormal,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					int numSamples, 
                                        double* intensity) {
	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);
	
        const int numThreads = (int) TaskScheduler::threadCount();

	smp::SamplerSet samplers(numThreads);

	parallel_for(int(0), int(numTriangles*numSources),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceIntensity(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, vertexNormal, 
				pathlengthLowerBound, pathlengthUpperBound, 
				numSamplesPerTriangle, intensity, numTriangles);
		}
	});
}



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
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int numSamples, 
                                        double* transient) {
	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);
        
	std::memset(transient, 0, numBins * numSources * sizeof(transient));
	
        const int numThreads = (int) TaskScheduler::threadCount();

	smp::SamplerSet samplers(numThreads);
        double* transient_refine; 
        if (refine_scale > 1) {
	    transient_refine = (double *) aligned_alloc(16,  numBins *sizeof(double) * numSources * refine_scale);
            std::memset(transient_refine, 0, numBins* sizeof(transient) *numSources * refine_scale);
        }
	double *transients = (double *) aligned_alloc(16, numThreads * numBins *sizeof(double) * numSources * refine_scale);
        if (transients == NULL) {
            printf("memory insufficient\n");
            return;
        }
        std::memset(transients, 0, numThreads * numBins* sizeof(transients) *numSources * refine_scale);

	parallel_for(int(0), int(numTriangles*numSources),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangle(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, vertexNormal, vertexAlbedo,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution/refine_scale,
				numSamplesPerTriangle, &transients[threadIndex * numSources * numBins * refine_scale], numTriangles);
		}
	});
        // collect transient
        if (refine_scale > 1) {

  	    parallel_for(int(0), int(numSources),[&](const range<int>& range) {
	        const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterSource = range.begin(); iterSource < range.end();															iterSource++) {
                     for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		         for (int iterBin = 0; iterBin < numBins * refine_scale; ++ iterBin) {
			     transient_refine[iterSource*numBins*refine_scale + iterBin] += transients[(iterThread * numSources + iterSource)* numBins * refine_scale +  iterBin];
			}
                     }
		}
	    });
            free(transients);
        } else {
  	    parallel_for(int(0), int(numSources),[&](const range<int>& range) {
	        const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterSource = range.begin(); iterSource < range.end();															iterSource++) {
                     for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		         for (int iterBin = 0; iterBin < numBins; ++ iterBin) {
			     transient[iterSource*numBins + iterBin] += transients[(iterThread * numSources + iterSource)* numBins + iterBin];
			}
                     }
		}
	    });

            free(transients);
            return;
        }

        
        double* convolution_kernal = (double*) aligned_alloc(16, numThreads*(4*refine_scale*sigma_bin+1)*sizeof(double));
        double* y = (double*) aligned_alloc(16, numThreads*(numBins*refine_scale + 4*refine_scale*sigma_bin)*sizeof(double));
        double sigma = pathlengthResolution*sigma_bin/2.355;
        double normalization = 1/sigma/embree::sqrt(2*M_PI)*pathlengthResolution/refine_scale;
        for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
            double t = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale/sigma;
            convolution_kernal[i] = embree::exp(-embree::sqr(t)/2)*normalization;
        }
        for (int i = 1; i < numThreads; ++i) {
            std::memcpy(&convolution_kernal[i*(4*refine_scale*sigma_bin+1)], convolution_kernal, (4*refine_scale*sigma_bin+1)*sizeof(double));
        } 

        parallel_for(int (0), int(numSources), [&](const range<int>& range) {
            const int threadIndex = (int) TaskScheduler::threadIndex();
            for (int iterS = range.begin(); iterS < range.end(); ++iterS) {
                double* new_y = &y[threadIndex*(numBins*refine_scale+4*refine_scale)];
                std::memset(new_y, 0, (numBins*refine_scale+4*refine_scale*sigma_bin)*sizeof(double));
                convolve(&convolution_kernal[threadIndex*(4*refine_scale*sigma_bin+1)], &transient_refine[iterS*numBins*refine_scale], new_y, 4*refine_scale*sigma_bin+1, numBins*refine_scale, numBins*refine_scale+4*refine_scale*sigma_bin, 0);

                for (int iterBin = 0; iterBin < numBins * refine_scale; ++iterBin) {
                    transient[iterS*numBins+iterBin/refine_scale] += new_y[iterBin + 2*refine_scale*sigma_bin];
                }
            }
        });       

        free(transient_refine);
        free(y);
        free(convolution_kernal);
}


void render_smoothed_vertex_gradients( int vertex_num,
					int numBins, 
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
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* gradient) {
	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);
	const int numThreads = (int) TaskScheduler::threadCount();
	smp::SamplerSet samplers(numThreads);
	
        double *gradients = (double *) aligned_alloc(16, numThreads * 3 * numBins * sizeof(gradients));
        if (gradients == NULL) {
            printf("grad memory insufficient\n");
            return;
        }
        std::memset(gradients, 0, numThreads * 3 * numVertices * sizeof(gradient));
        double* weighting_kernal = (double*) aligned_alloc(16, numThreads*(4*refine_scale*sigma_bin+1)*sizeof(double));
        double sigma = pathlengthResolution*sigma_bin/2.355;
        double sigma_square = sigma*sigma;
        double normalization = 1/sigma/embree::sqrt(2*M_PI)*pathlengthResolution/refine_scale;
        for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
            double t = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale/sigma;
            weighting_kernal[i] = embree::exp(-embree::sqr(t)/2)*normalization;
        }
        for (int i = 1; i < numThreads; ++i) {
            std::memcpy(&weighting_kernal[i*(4*refine_scale*sigma_bin+1)], weighting_kernal, (4*refine_scale*sigma_bin+1)*sizeof(double));
        } 
	parallel_for(int(0), int(numTriangles*numSources),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangleVertexGradient(vertex_num, iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, 
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, &gradients[threadIndex*3*numBins], 
                                numTriangles, numVertices, 
                                refine_scale, sigma_bin,
                                &weighting_kernal[threadIndex*(4*refine_scale*sigma_bin+1)], sigma_square);
                }
	
	});
	for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		for (int32_t iterD = 0; iterD < numBins * 3 ; ++iterD) {
	  		gradient[iterD] += gradients[iterThread * 3 * numBins + iterD]/numSources;
		}
	}
        
        free(gradients);
        free(weighting_kernal);
}

double render_smoothed_gradients_albedo(int numBins, 
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
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference) {
	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);
	const int numThreads = (int) TaskScheduler::threadCount();
	smp::SamplerSet samplers(numThreads);
	
        double *gradients = (double *) aligned_alloc(16, numThreads * sizeof(double));
        if (gradients == NULL) {
            printf("grad memory insufficient\n");
            return 0;
        }
        std::memset(gradients, 0, numThreads * sizeof(double));
        double* weighting_kernal = (double*) aligned_alloc(16, numThreads*(4*refine_scale*sigma_bin+1)*sizeof(double));
        double sigma = pathlengthResolution*sigma_bin/2.355;
        double sigma_square = sigma*sigma;
        double normalization = 1/sigma/embree::sqrt(2*M_PI)*pathlengthResolution/refine_scale;
        for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
            double t = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale/sigma;
            weighting_kernal[i] = embree::exp(-embree::sqr(t)/2)*normalization;
        }
        for (int i = 1; i < numThreads; ++i) {
            std::memcpy(&weighting_kernal[i*(4*refine_scale*sigma_bin+1)], weighting_kernal, (4*refine_scale*sigma_bin+1)*sizeof(double));
        } 
	parallel_for(int(0), int(numTriangles*numSources),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangleGradientAlbedo(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, vertexNormal, vertexAlbedo,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, difference, &gradients[threadIndex], 
                                numTriangles, numVertices, 
                                refine_scale, sigma_bin,
                                &weighting_kernal[threadIndex*(4*refine_scale*sigma_bin+1)], sigma_square);
                }
	
	});
	double gradient = 0;
	for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
	    gradient += gradients[iterThread];
	}
	gradient /= numSources;
        
        free(gradients);
        free(weighting_kernal);
        return gradient;
}


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
                                        const float* vertexNormal,
					const float *vertexAlbedo,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* difference,
                                        double* gradient,
                                        int testing_flag) {
	const int32_t numSamplesPerTriangle = 1 + ((numSamples - 1) / numTriangles);
	const int numThreads = (int) TaskScheduler::threadCount();
	smp::SamplerSet samplers(numThreads);
	
        double *gradients = (double *) aligned_alloc(16, numThreads * 3 * numVertices * sizeof(gradients));
        if (gradients == NULL) {
            printf("grad memory insufficient\n");
            return;
        }
        std::memset(gradients, 0, numThreads * 3 * numVertices * sizeof(gradient));
        double* weighting_kernal = (double*) aligned_alloc(16, numThreads*(4*refine_scale*sigma_bin+1)*sizeof(double));
        double sigma = pathlengthResolution*sigma_bin/2.355;
        double sigma_square = sigma*sigma;
        double normalization = 1/sigma/embree::sqrt(2*M_PI)*pathlengthResolution/refine_scale;
        for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
            double t = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale/sigma;
            weighting_kernal[i] = embree::exp(-embree::sqr(t)/2)*normalization;
        }
        for (int i = 1; i < numThreads; ++i) {
            std::memcpy(&weighting_kernal[i*(4*refine_scale*sigma_bin+1)], weighting_kernal, (4*refine_scale*sigma_bin+1)*sizeof(double));
        } 
	parallel_for(int(0), int(numTriangles*numSources),[&](const range<int>& range) {
	    	const int threadIndex = (int) TaskScheduler::threadIndex();
	 	for (int iterTriangle_source = range.begin(); iterTriangle_source < range.end();															iterTriangle_source++) {
			streamedRayTraceTriangleGradient(iterTriangle_source, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, vertexNormal, vertexAlbedo,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, difference, &gradients[threadIndex*3*numVertices], 
                                numTriangles, numVertices, 
                                refine_scale, sigma_bin,
                                &weighting_kernal[threadIndex*(4*refine_scale*sigma_bin+1)], sigma_square, testing_flag);
                }
	
	});
	for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		for (int32_t iterD = 0; iterD < numVertices * 3 ; ++iterD) {
	  		gradient[iterD] += gradients[iterThread * 3 * numVertices + iterD]/numSources;
		}
	}
        
        free(gradients);
        free(weighting_kernal);
}

void streamedRayTraceTriangleGradientAlbedo(int32_t triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
                                        const float * normals,
					const float *albedoes,
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
                                        double sigma_square) {
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

	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal)/2;
	faceNormal /= faceArea*2;

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
                                float cos_theta2 = dot(originNormal, rays[iterSample].dir);
                                float cos_theta3 = dot(normal, -rays[iterSample].dir);
                                if (cos_theta2 < 0) cos_theta2 = 0;
                                if (cos_theta3 < 0) cos_theta3 = 0;
				float formFactor = cos_theta2 * cos_theta3/ halfLength
							     / halfLength;
                                double g0 = formFactor *formFactor;
                                
                                // compute gradient 

                                double delta_length, g;
                                int32_t bin;
                                for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
                                    delta_length = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale;
				    bin = (int32_t) embree::floor((2.0f * halfLength + delta_length - pathlengthLowerBound)
														/ pathlengthResolution);
                                    g = g0 * weighting_kernal[i] * (-2) * difference[sourceIndex*numBins + bin];                                    
                                    gradient[0] += (double) (faceArea * g)/ (double) numSamples;
                                }

			}
		}
	}
        free(rays);
}

void streamedRayTraceTriangleVertexGradient(int vertex_num,
					int32_t triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
					float pathlengthLowerBound,
					float pathlengthUpperBound,
					float pathlengthResolution,
					int32_t numSamples, 
                                        double* gradient,
					int numTriangle,
                                        int numVertices,
                                        int refine_scale,
                                        int sigma_bin,
                                        double* weighting_kernal, 
                                        double sigma_square) {
	Ray *rays = (Ray *) aligned_alloc(16, numSamples  * sizeof(Ray));
        
	int triangleIndex = triangle_source_index % numTriangle;
        int sourceIndex = triangle_source_index / numTriangle;

	Vec3fa origin(originD[3*sourceIndex], originD[3*sourceIndex + 1], originD[3*sourceIndex + 2]);
	Vec3fa originNormal(originNormalD[3*sourceIndex], originNormalD[3*sourceIndex + 1], originNormalD[3*sourceIndex + 2]);

	int32_t v1Ind = triangles[3 * triangleIndex];
	int32_t v2Ind = triangles[3 * triangleIndex + 1];
	int32_t v3Ind = triangles[3 * triangleIndex + 2];

	if (v1Ind != vertex_num && v2Ind != vertex_num && v3Ind != vertex_num) {
            return;
        }

	Vec3fa v1(vertices[3 * v1Ind], vertices[3 * v1Ind + 1], vertices[3 * v1Ind + 2]);
	Vec3fa v2(vertices[3 * v2Ind], vertices[3 * v2Ind + 1], vertices[3 * v2Ind + 2]);
	Vec3fa v3(vertices[3 * v3Ind], vertices[3 * v3Ind + 1], vertices[3 * v3Ind + 2]);

	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal)/2;
	faceNormal /= faceArea*2;

	Vec3fa n1(0, 0, 1), n2(0, 0, 1), n3(0, 0, 1);

	float a1(1), a2(1), a3(1);

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

	for (int32_t iterSample = 0; iterSample < numSamples; ++iterSample) {

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
                                float cos_theta2 = dot(originNormal, rays[iterSample].dir);
                                float cos_theta3 = dot(normal, -rays[iterSample].dir);
                                if (cos_theta2 < 0) cos_theta2 = 0;
                                if (cos_theta3 < 0) cos_theta3 = 0;
				float formFactor = cos_theta2 * cos_theta3/ halfLength
							     / halfLength;
                                double intensity = albedo * formFactor *formFactor;
                                
                                // compute gradient 
				Vec3fa t1 = 2 * albedo * cos_theta2 * cos_theta3 * (originNormal * cos_theta3 - normal * cos_theta2 + 4 * (-rays[iterSample].dir) * cos_theta2 * cos_theta3);
                                t1 /= embree::pow(halfLength,5);

                                Vec3fa t2 = normal*intensity;

                                Vec3fa gn(0,0,0);
                                gn = - 2 * albedo * rays[iterSample].dir * cos_theta3 * cos_theta2 * cos_theta2;
                                gn /= embree::pow(halfLength, 4);
                                float cos_tmp = dot(gn, normal);
                                gn -= normal*cos_tmp;

                                t2 = (t2 + gn)/(2*faceArea);
                                 
                                double delta_length;
                                Vec3fa gauss_grad, e, g;
                                int32_t bin;
                                for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
                                    delta_length = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale;
                                    gauss_grad = delta_length/sigma_square*2*rays[iterSample].dir;                     
				    bin = (int32_t) embree::floor((2.0f * halfLength + delta_length - pathlengthLowerBound)
														/ pathlengthResolution);

			            if (vertex_num == v1Ind) {
                                        e = v3 - v2;
                                        g = (t1 + gauss_grad * intensity)*u + cross(t2,e);
                                        g *= weighting_kernal[i];                                    
                                    }
                                    else if (vertex_num == v2Ind) {
                                      e = v1-v3;
                                      g = (t1 + gauss_grad * intensity)*v + cross(t2,e);
                                      g *= weighting_kernal[i];                                    
                                    }
                                    else {

                                      e = v2-v1;
                                      g = (t1 + gauss_grad * intensity)*w + cross(t2,e);
                                      g *= weighting_kernal[i];                                    
                                    }
                                    gradient[3 * bin] += (double) (faceArea * g[0])/ (double) numSamples;
                                    gradient[3 * bin + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                    gradient[3 * bin + 2] += (double) (faceArea * g[2])/ (double) numSamples;
                                }

			}
		}
	}
        free(rays);
}


void streamedRayTraceTriangleGradient(int32_t triangle_source_index,
					RTCDevice g_device,
					RTCScene g_scene,
					smp::Sampler &sampler,
					const float *originD,
					const float *originNormalD,
					const float *vertices,
					const int32_t *triangles,
                                        const float * normals,
					const float *albedoes,
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
                                        double sigma_square,
					int testing_flag) {
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

	Vec3fa faceNormal = cross(v2 - v1, v3 - v1);
	float faceArea = length(faceNormal)/2;
	faceNormal /= faceArea*2;

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
                                float cos_theta2 = dot(originNormal, rays[iterSample].dir);
                                float cos_theta3 = dot(normal, -rays[iterSample].dir);
                                if (cos_theta2 < 0) cos_theta2 = 0;
                                if (cos_theta3 < 0) cos_theta3 = 0;
				float formFactor = cos_theta2 * cos_theta3/ halfLength
							     / halfLength;
                                double intensity = albedo * formFactor *formFactor;
                                
                                // compute gradient 
				Vec3fa t1 = 2 * albedo * cos_theta2 * cos_theta3 * (originNormal * cos_theta3 - normal * cos_theta2 + 4 * (-rays[iterSample].dir) * cos_theta2 * cos_theta3);
                                t1 /= embree::pow(halfLength,5);

                                Vec3fa t2 = normal*intensity;

                                Vec3fa gn(0,0,0);
                                if (testing_flag == 0 && normals != nullptr) {
                                    gn = - 2 * albedo * rays[iterSample].dir * cos_theta3 * cos_theta2 * cos_theta2;
                                    gn /= embree::pow(halfLength, 4);
                                    float cos_tmp = dot(gn, normal);
                                    gn -= normal*cos_tmp;
                                }

                                t2 = (t2 + gn)/(2*faceArea);

                                 
                                double delta_length;
                                Vec3fa gauss_grad, e, g;
                                int32_t bin;
                                for (int i = 0; i < 4*refine_scale*sigma_bin+1; ++i) {
                                    delta_length = (-2*refine_scale*sigma_bin+i)*pathlengthResolution/refine_scale;
                                    gauss_grad = delta_length/sigma_square*2*rays[iterSample].dir;                     
				    bin = (int32_t) embree::floor((2.0f * halfLength + delta_length - pathlengthLowerBound)
														/ pathlengthResolution);
                                    e = v3 - v2;
                                    g = (t1 + gauss_grad * intensity)*u + cross(t2,e);
                                    g *= weighting_kernal[i];                                    
				    g *= (-2)*difference[sourceIndex * numBins + bin];
                                    gradient[3 * v1Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                    gradient[3 * v1Ind + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                    gradient[3 * v1Ind + 2] += (double) (faceArea * g[2])/ (double) numSamples;
                                   
                                    e = v1-v3;
                                    g = (t1 + gauss_grad * intensity)*v + cross(t2,e);
                                    g *= weighting_kernal[i];                                    
			   	    g *= (-2)*difference[sourceIndex * numBins + bin];
                                    gradient[3 * v2Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                    gradient[3 * v2Ind + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                    gradient[3 * v2Ind + 2] += (double) (faceArea * g[2])/ (double) numSamples;

                                    e = v2-v1;
                                    g = (t1 + gauss_grad * intensity)*w + cross(t2,e);
                                    g *= weighting_kernal[i];                                    
				    g *= (-2)*difference[sourceIndex * numBins + bin];
                                    gradient[3 * v3Ind] += (double) (faceArea * g[0])/ (double) numSamples;
                                    gradient[3 * v3Ind + 1] += (double) (faceArea * g[1])/ (double) numSamples;
                                    gradient[3 * v3Ind + 2] += (double) (faceArea * g[2])/ (double) numSamples;

                                }

			}
		}
	}
        free(rays);
}

