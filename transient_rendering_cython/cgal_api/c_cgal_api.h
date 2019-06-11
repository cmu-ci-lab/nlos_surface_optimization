#ifndef C_CGAL_API_H
#define C_CGAL_API_H
void keep_largest_connected_component(float* vertex, int& num_vertices, int* face, int& num_faces);
void per_vertex_normal(float* vertex, int num_vertices, int* face, int num_faces, float* vn);
int find_convex_hull(float* vertex, int num_vertices, float* hull_location);
void isotropic_remeshing(float* vertex, int& num_vertices, int* face, int& num_faces, double target_edge_length, int nb_iter, float* new_vertex, int* new_face, int prepared_vertices, int prepared_face);
void border_vertex(float* vertex, int num_vertices, int* face, int num_faces, int* result);
void face_affinity(float* vertex, int num_vertices, int* face, int num_faces, int* f_affinity);
#endif
