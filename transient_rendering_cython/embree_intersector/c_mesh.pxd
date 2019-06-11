cdef extern from "c_mesh.h":
  cdef cppclass Mesh:
    Mesh()
    Mesh(float*, int, int*, int)
    void test()
    void embree3_tbb_line_intersection(float*, float*, int, float*)
    void embree3_tbb_short_line_intersection(float*, float*, int, float*)
    void barycentric_to_world(float*, int, float*)
    int get_vertex_num()
    int get_face_num()
    void set_vn(float*)
    void set_fn_and_face_area(float*, float*)
