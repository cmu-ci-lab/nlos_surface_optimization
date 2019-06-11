/*
 * intersectors//intersectors/stratified_transient_raytracer/stratifiedTransientRenderer.cpp/stratifiedTransientRenderer.cpp
 *
 *  Created on: Aug 24, 2018
 *      Author: igkiou
 */

#include "stratifiedTransientRenderer.h"

/* include embree API */
#include <embree3/rtcore.h>
#include <common/math/math.h>
#include <common/math/vec.h>
#include <common/math/affinespace.h>
#include <common/core/ray.h>
#include <tasking/taskscheduler.h>
#include <algorithms/parallel_for.h>
#include "sampler.h"

using namespace embree;

/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

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
					double *transient) {

	Vec3fa origin(originD[0], originD[1], originD[2]);
	Vec3fa originNormal(originNormalD[0], originNormalD[1], originNormalD[2]);

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
	faceNormal /= 2 * faceArea;

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

	/*
	 * TODO: Since all of the rays here are very coherent, replace with
	 * coherent streaming.
	 */
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
		Vec3fa direction = point - origin;
		float halfLength = length(direction);
		direction = direction / halfLength;

		/* initialize ray */
		Ray ray(origin, direction, zero, inf);

		/* intersect ray with scene */
		rtcIntersect1(g_scene, &context, RTCRayHit_(ray));

		if ((ray.geomID != RTC_INVALID_GEOMETRY_ID) && (ray.primID == triangleIndex)
				&& (halfLength <= pathlengthUpperBound / 2.0f)
				&& (halfLength >= pathlengthLowerBound / 2.0f)) {
			Vec3fa normal(faceNormal);
			if (normals != nullptr) {
				normal = u * n1 + v * n2 + w * n3;
			}
			float albedo(1);
			if (albedoes != nullptr) {
				albedo = u * a1 + v * a2 + w * a3;
			}
			float formFactor = - dot(normal, direction)
								* dot(originNormal, direction)
								/ halfLength
								/ halfLength;

			int32_t bin = (int32_t) embree::floor((2.0f * halfLength - pathlengthLowerBound)
													/ pathlengthResolution);
			transient[bin] += (double) (faceArea * albedo * formFactor * formFactor)
							/ (double) numSamples;
		}
	}
}


//void render_transient(float* originD, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlegnthResolution, float* normalsD, float* albedosD, double* transient, double *pathlengths) {
void render_transient(float* originD, float* normalD, float* verticesD, int numVertices, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* transient, double *pathlengths) {
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
//	const int32_t adjustedNumSamples = numSamplesPerTriangle * numTriangles;

	const int32_t numBins = (int32_t) embree::ceil((pathlengthUpperBound - pathlengthLowerBound)
													/ pathlengthResolution);

	for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
		pathlengths[iterBin] =
			(double) (pathlengthLowerBound + iterBin * pathlengthResolution);
	}
	
	const int numThreads = (int) TaskScheduler::threadCount();
	smp::SamplerSet samplers(numThreads);
	double *transients = (double *) aligned_alloc(16, numThreads * numBins *sizeof(double));
	//double *transients = (double *) alignedMalloc(numThreads * numBins *sizeof(double), 16);
	std::memset(transients, 0, numThreads * numBins * sizeof(double));

	/* render triangles */
	parallel_for(size_t(0), size_t(numTriangles),[&](const range<size_t>& range) {
		const int threadIndex = (int) TaskScheduler::threadIndex();
		for (int32_t iterTriangle = range.begin(); iterTriangle < range.end();
																iterTriangle++) {
			rayTraceTriangle(iterTriangle, g_device, g_scene, samplers[threadIndex],
				originD, normalD, verticesD, trianglesD, nullptr, nullptr,
				pathlengthLowerBound, pathlengthUpperBound, pathlengthResolution,
				numSamplesPerTriangle, &transients[threadIndex * numBins]);
  		}
	});

	std::memset(transient, 0, numBins * sizeof(transient));
	for (int iterThread = 0; iterThread < numThreads; ++iterThread) {
		for (int32_t iterBin = 0; iterBin < numBins; ++iterBin) {
			transient[iterBin] += transients[iterThread * numBins + iterBin];
		}
	}

	/*
	 * TODO: Free alignedMalloced memory.
	 */
	//alignedFree(transients);
        
        free(transients);
	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}
