#include "c_el_topo_api.h"

#include <eltopo.h>
#include <subdivisionscheme.h>

#include <cstring>
#include <stdlib.h>
#include <stdio.h>

void el_topo_remesh(double* v, int& v_num, int* f, int& f_num, double* new_v, int new_v_num, int* new_f, int new_f_num, double edge_length) {
    ElTopoMesh inputs;
    inputs.num_vertices = v_num;
    inputs.vertex_locations = v;

    inputs.num_triangles = f_num;
    inputs.triangles = f;

    inputs.vertex_masses = new double [inputs.num_vertices];    
    std::fill(inputs.vertex_masses, &inputs.vertex_masses[inputs.num_vertices], 1.0);


    ElTopoGeneralOptions general_options;
    general_options.m_verbose = true;
    general_options.m_collision_safety = false;
    general_options.m_proximity_epsilon = 0.0001;

    ElTopoStaticOperationsOptions options;
    options.m_min_edge_length = edge_length * 0.99;
    options.m_max_edge_length = edge_length * 1.01;
    options.m_max_volume_change = 0.01;
    options.m_min_triangle_angle = 30;
    options.m_max_triangle_angle = 90;
    options.m_use_curvature_when_splitting = true;
    options.m_use_curvature_when_collapsing = true;

    options.m_min_curvature_multiplier = 1.0;
    options.m_max_curvature_multiplier = 1.0;
    options.m_allow_vertex_movement = true;

    options.m_merge_proximity_epsilon = edge_length/10;

    options.m_allow_non_manifold = false;
    options.m_allow_topology_changes = true;
    options.m_perform_improvement = true;
    options.m_subdivision_scheme = new ButterflyScheme();
 
    ElTopoMesh outputs;
    ElTopoDefragInformation defrag_info;

    el_topo_static_operations( &inputs, &general_options, &options, &defrag_info, &outputs );

    v_num = outputs.num_vertices;
    f_num = outputs.num_triangles;
    if (v_num > new_v_num || f_num > new_f_num)
        return;

    for (int i = 0; i < v_num; ++i) {
        new_v[3*i] = outputs.vertex_locations[3*i];
        new_v[3*i+1] = outputs.vertex_locations[3*i+1];
        new_v[3*i+2] = outputs.vertex_locations[3*i+2];
    }
    for (int i = 0; i < f_num; ++i) {
        new_f[3*i] = outputs.triangles[3*i];
        new_f[3*i+1] = outputs.triangles[3*i+1];
        new_f[3*i+2] = outputs.triangles[3*i+2];
    }
    delete [] inputs.vertex_masses;
    delete [] outputs.vertex_locations;
    delete [] outputs.triangles;
    delete [] outputs.vertex_masses;
}



void el_topo_gradient(double* v, int v_num, int* f, int f_num, double* new_v) {
    ElTopoMesh inputs;
    inputs.num_vertices = v_num;
    inputs.vertex_locations = v;

    inputs.num_triangles = f_num;
    inputs.triangles = f;

    inputs.vertex_masses = new double [inputs.num_vertices];    
    std::fill(inputs.vertex_masses, &inputs.vertex_masses[inputs.num_vertices], 1.0);

    ElTopoGeneralOptions general_options;
    general_options.m_verbose = 0;
    general_options.m_collision_safety = true;
    ElTopoIntegrationOptions options;
    options.m_dt = 1;

    double* out_vertex_locations;
    double out_dt;
    el_topo_integrate(&inputs, new_v, &general_options, &options, &out_vertex_locations, &out_dt);

    memcpy(new_v, out_vertex_locations, 3*inputs.num_vertices*sizeof(double));

    el_topo_free_integrate_results(out_vertex_locations);
    delete [] inputs.vertex_masses;
}

