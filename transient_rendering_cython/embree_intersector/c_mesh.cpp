#include "c_mesh.h"

#include <embree3/rtcore.h>
#include <common/math/math.h>
#include <common/math/vec.h>
#include <common/math/affinespace.h>
#include <common/core/ray.h>
#include <tasking/taskscheduler.h>
#include <algorithms/parallel_for.h>

#include <stdio.h>

using namespace embree;

/* vertex and triangle layout */
struct Vertex   { float x,y,z,r;  }; // FIXME: rename to Vertex4f
struct Triangle { int v0, v1, v2; };

Mesh::Mesh() {
  vn_available = false;
}

Mesh::Mesh(float* vertices, int vertice_num, int* faces, int face_num) {
  this->verticesD = vertices;
  this->numVertices = vertice_num;
  this->trianglesD = faces;
  this->numTriangles = face_num;
  this->vn_available = false;
}

void Mesh::set_vn(float* vn) {
  this->vn = vn;
  this->vn_available = true;
}


void Mesh::set_fn_and_face_area(float* fn, float* face_area) {
  this->fn = fn;
  this->face_area = face_area;
}

int Mesh::get_vertex_num() {
  return numVertices; 
}

int Mesh::get_face_num() {
  return numTriangles; 
}

void Mesh::test() {
  printf("vertices\n");
  for (int i = 0; i < numVertices; ++i) {
      printf("%f %f %f\n", verticesD[3*i], verticesD[3*i+1], verticesD[3*i+2] );
  }
  printf("faces\n");
  for (int i=0; i < numTriangles; ++i) {
      printf("%d %d %d \n", trianglesD[3*i], trianglesD[3*i+1], trianglesD[3*i+2]);
  }
}

void Mesh::embree3_tbb_line_intersection(float* originsD, float* directionsD, int num_ray, float* intersect) {
    const size_t numRays = (size_t) num_ray;
    
    /* start embree device */
    RTCDevice g_device = rtcNewDevice(nullptr);

    /* scene data */
    RTCScene g_scene = rtcNewScene(g_device);
    /* create a triangulated plane with 2 triangles and 4 vertices */
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

    /* render image */
    parallel_for(size_t(0), size_t(numRays),[&](const range<size_t>& range) {
	const int threadIndex = (int) TaskScheduler::threadIndex();
	for (size_t iterRay = range.begin(); iterRay < range.end(); iterRay++) {
		rayTrace((int) iterRay, threadIndex, g_device, g_scene, originsD, directionsD, intersect);
	}
    });
    rtcReleaseScene(g_scene); g_scene = nullptr;
    rtcReleaseDevice(g_device);
}

void Mesh::embree3_tbb_short_line_intersection(float* originsD, float* directionsD, int num_ray, float* intersect) {
    const size_t numRays = (size_t) num_ray;
    
    /* start embree device */
    RTCDevice g_device = rtcNewDevice(nullptr);

    /* scene data */
    RTCScene g_scene = rtcNewScene(g_device);
    /* create a triangulated plane with 2 triangles and 4 vertices */
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

    /* render image */
    parallel_for(size_t(0), size_t(numRays),[&](const range<size_t>& range) {
	const int threadIndex = (int) TaskScheduler::threadIndex();
	for (size_t iterRay = range.begin(); iterRay < range.end(); iterRay++) {
		rayTrace_short((int) iterRay, threadIndex, g_device, g_scene, originsD, directionsD, intersect);
	}
    });
    rtcReleaseScene(g_scene); g_scene = nullptr;
    rtcReleaseDevice(g_device);
}

void Mesh::barycentric_to_world(float* barycoord, int num_ray, float* intersection_p) {
    const size_t numRays = (size_t) num_ray;
   
    parallel_for(size_t(0), size_t(numRays),[&](const range<size_t>& range) {
	const int threadIndex = (int) TaskScheduler::threadIndex();
	for (size_t iterRay = range.begin(); iterRay < range.end(); iterRay++) {
		coord_conversion((int) iterRay, threadIndex, barycoord, intersection_p);
	}
    });

}

void Mesh::coord_conversion(int iterRay, int threadIndex, float* barycoord, float* intersection_p) {  
  int fid = (int) barycoord[iterRay*3];
  
  if (fid < 0)
     return;
  float u = barycoord[iterRay*3 + 1]; 
  float v = barycoord[iterRay*3 + 2];
  
  int v1 = trianglesD[fid*3];
  int v2 = trianglesD[fid*3+1];
  int v3 = trianglesD[fid*3+2];
  

  for (int i = 0; i < 3; ++i) {
     intersection_p[iterRay*3 + i] = (1-u-v)*verticesD[v1*3+i] + u*verticesD[v2*3+i] + v*verticesD[v3*3+i]; 
  }  
}


void Mesh::rayTrace(int iterRay,
			int threadIndex,
			RTCDevice g_device,
			RTCScene g_scene,
			const float *originsD,
			const float *directionsD,
			float *intersect) {

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	/* initialize ray */
	Ray ray(
		Vec3fa(originsD[iterRay * 3], originsD[iterRay * 3 + 1], originsD[iterRay * 3 + 2]),
		Vec3fa(directionsD[iterRay * 3], directionsD[iterRay * 3 + 1], directionsD[iterRay * 3 + 2]),
		zero, inf);

	/* intersect ray with scene */
	rtcIntersect1(g_scene, &context, RTCRayHit_(ray));

	if (ray.geomID == RTC_INVALID_GEOMETRY_ID) {
		intersect[iterRay*3] = (float) -1;
	} else {
		intersect[iterRay*3] = ray.primID ;
	  	intersect[iterRay*3 + 1] = ray.u;
		intersect[iterRay*3 + 2] = ray.v;
        }
}
void Mesh::rayTrace_short(int iterRay,
			int threadIndex,
			RTCDevice g_device,
			RTCScene g_scene,
			const float *originsD,
			const float *directionsD,
			float *intersect) {

	RTCIntersectContext context;
	rtcInitIntersectContext(&context);
	/* initialize ray */
	Ray ray(
		Vec3fa(originsD[iterRay * 3], originsD[iterRay * 3 + 1], originsD[iterRay * 3 + 2]),
		Vec3fa(directionsD[iterRay * 3], directionsD[iterRay * 3 + 1], directionsD[iterRay * 3 + 2]),
		zero, inf);

	/* intersect ray with scene */
	rtcIntersect1(g_scene, &context, RTCRayHit_(ray));

	if (ray.geomID == RTC_INVALID_GEOMETRY_ID) {
		intersect[iterRay] = (float) -1;
	} else {
		intersect[iterRay] = ray.primID ;
        }
}

