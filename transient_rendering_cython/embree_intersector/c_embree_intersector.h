#include <embree3/rtcore.h>

void coord_conversion(int, int, float*, int*, float*, float*);
void barycentric_to_world(float*, int*, float*, int, float*);

void rayTrace(int, int, RTCDevice, RTCScene, const float*, const float*, float*); 
void rayTrace_short(int, int, RTCDevice, RTCScene, const float*, const float*, float*); 
void embree3_tbb_line_intersection(float*, float*, int, float*, int, int*, int, float*);
void embree3_tbb_short_line_intersection(float*, float*, int, float*, int, int*, int, float*);
