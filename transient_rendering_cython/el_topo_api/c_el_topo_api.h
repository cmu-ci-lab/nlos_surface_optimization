#ifndef C_EL_TOPO_API_H
#define C_EL_TOPO_API_H
void el_topo_gradient(double* v, int v_num, int* f, int f_num, double* new_v);
void el_topo_remesh(double* v, int& v_num, int* f, int& f_num, double* new_v, int new_v_num, int* new_f, int new_f_num, double edge_length);
#endif
