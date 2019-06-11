#ifndef MESH_H
#define MESH_H

#include <embree3/rtcore.h>


class Mesh {
    float* verticesD;
    int numVertices;
    int* trianglesD;
    int numTriangles;
    float* vn;
    bool vn_available;
    float* fn;
    float* face_area;
    
    void coord_conversion(int, int, float*, float*);

    void rayTrace(int, int, RTCDevice, RTCScene, const float*, const float*, float*); 
    void rayTrace_short(int, int, RTCDevice, RTCScene, const float*, const float*, float*); 

public:
    Mesh();
    Mesh(float*, int, int*, int);
    void test();
    
    void embree3_tbb_line_intersection(float*, float*, int, float*);
    void embree3_tbb_short_line_intersection(float*, float*, int, float*);
    void barycentric_to_world(float*, int, float*);
    int get_vertex_num();
    int get_face_num();
    void set_vn(float*);
    void set_fn_and_face_area(float*, float*);
};

#endif
