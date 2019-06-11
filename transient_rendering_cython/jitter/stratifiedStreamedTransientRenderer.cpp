/*
 * stratifiedStreamedTransientRenderer.cpp
 *
 *  Created on: Aug 28, 2018
 *      Author: igkiou
 */
#include "stratifiedStreamedTransientRenderer.h"

/* include embree API */
#include <embree3/rtcore.h>

#include "transient_and_gradient.h"

using namespace embree;
/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };


void streamed_render_transient(float* originD, int numSources, float* normalD, float* verticesD, int numVertices, float* vertexNormal, float* vertexAlbedo, int* trianglesD, int numTriangles, int numSamples, float pathlengthLowerBound, float pathlengthUpperBound, float pathlengthResolution, double* weight, int weight_offset, int weight_length, double* transient, double *pathlengths) {
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
                                        numSources, 
                                        numTriangles, 
                                        g_device, 
                                        g_scene, 
                                        originD, 
                                        normalD, 
					verticesD,
					trianglesD,
					vertexNormal,
					vertexAlbedo,
					pathlengthLowerBound,
					pathlengthUpperBound,
					pathlengthResolution,
					weight,
					weight_offset,
					weight_length,
					numSamples, 
                                        transient);


	rtcReleaseScene(g_scene); g_scene = nullptr;
	rtcReleaseDevice(g_device);
}
