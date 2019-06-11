#include "c_cgal_api.h"

#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include <CGAL/convex_hull_2.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>

#include <CGAL/boost/graph/iterator.h>

#include <boost/foreach.hpp>
#include <boost/function_output_iterator.hpp>

#include <fstream>
#include <stdio.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;
typedef std::vector<Point_2> Points;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector;
typedef K::Compare_dihedral_angle_3 Compare_dihedral_angle_3;

typedef CGAL::Surface_mesh<Point_3> Mesh;

typedef Mesh::Vertex_index V_ID;
typedef Mesh::Face_index F_ID;
typedef Mesh::Face_iterator F_IT;
typedef Mesh::size_type size_type;
typedef boost::graph_traits<Mesh>::halfedge_descriptor halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor     edge_descriptor;
typedef boost::graph_traits<Mesh>::vertex_descriptor vertex_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor   face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;



template <typename G>
struct Constraint : public boost::put_get_helper<bool,Constraint<G> >
{
  typedef boost::readable_property_map_tag      category;
  typedef bool                                  value_type;
  typedef bool                                  reference;
  typedef edge_descriptor                       key_type;
  Constraint()
    :g_(NULL)
  {}
  Constraint(G& g, double bound) 
    : g_(&g), bound_(bound)
  {}
  bool operator[](edge_descriptor e) const
  {
    const G& g = *g_;
    return compare_(g.point(source(e, g)),
                    g.point(target(e, g)),
                    g.point(target(next(halfedge(e, g), g), g)),
                    g.point(target(next(opposite(halfedge(e, g), g), g), g)),
                   bound_) == CGAL::SMALLER;
  }
  const G* g_;
  Compare_dihedral_angle_3 compare_;
  double bound_;
};




struct halfedge2edge
{
  halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges)
    : m_mesh(m), m_edges(edges)
  {}
  void operator()(const halfedge_descriptor& h) const
  {
    m_edges.push_back(edge(h, m_mesh));
  }
  const Mesh& m_mesh;
  std::vector<edge_descriptor>& m_edges;
};

void keep_largest_connected_component(float* vertex, int& num_vertices, int* face, int& num_faces) {
  const double bound = std::cos(0.75*CGAL_PI);

  Mesh mesh;
  for (int i = 0; i < num_vertices; ++i) {
    mesh.add_vertex(Point_3(vertex[3*i], vertex[3*i+1], vertex[3*i+2]));
  }
  for (int i = 0; i < num_faces; ++i) {
    mesh.add_face(V_ID(face[3*i]), V_ID(face[3*i+1]), V_ID(face[3*i+2]));
  }


  size_t num = PMP::keep_largest_connected_components(mesh, 1);
  if (num == 0)
      return;

  std::vector<int> reindex;
  reindex.resize(mesh.num_vertices());

  int n = 0;
  BOOST_FOREACH(V_ID vd, mesh.vertices()) {
    auto p = mesh.point(vd);
    vertex[3*n] = p.x();
    vertex[3*n+1] = p.y();
    vertex[3*n+2] = p.z();
    reindex[vd] = n++;
  }
  num_vertices = n;

  int n_f = 0;
  BOOST_FOREACH(F_ID fd, mesh.faces()) {
    int fj = 0;
    BOOST_FOREACH(V_ID vd, CGAL::vertices_around_face(mesh.halfedge(fd), mesh)) {
      face[3*n_f+fj] = reindex[vd];
      ++fj;
    }
    ++n_f;
  }
  num_faces = n_f;
}


void per_vertex_normal(float* vertex, int num_vertices, int* face, int num_faces, float* vn) {
  Mesh mesh;
  for (int i = 0; i < num_vertices; ++i) {
    mesh.add_vertex(Point_3(vertex[3*i], vertex[3*i+1], vertex[3*i+2]));
  }
  for (int i = 0; i < num_faces; ++i) {
    mesh.add_face(V_ID(face[3*i]), V_ID(face[3*i+1]), V_ID(face[3*i+2]));
  }

  Mesh::Property_map<vertex_descriptor, Vector> vnormals =
    mesh.add_property_map<vertex_descriptor, Vector>
      ("1v:normals", CGAL::NULL_VECTOR).first;


  CGAL::Polygon_mesh_processing::compute_vertex_normals(mesh, vnormals,
      CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh.points()).geom_traits(K()));

 
  BOOST_FOREACH(vertex_descriptor vd, vertices(mesh)){
    vn[3*vd] = vnormals[vd].x();
    vn[3*vd+1] = vnormals[vd].y();
    vn[3*vd+2] = vnormals[vd].z();
  } 

}


void face_affinity(float* vertex, int num_vertices, int* face, int num_faces, int* f_affinity) {
  Mesh mesh;
  for (int i = 0; i < num_vertices; ++i) {
    mesh.add_vertex(Point_3(vertex[3*i], vertex[3*i+1], vertex[3*i+2]));
  }
  for (int i = 0; i < num_faces; ++i) {
    mesh.add_face(V_ID(face[3*i]), V_ID(face[3*i+1]), V_ID(face[3*i+2]));
  }
  
  int n_f = 0;
  BOOST_FOREACH(F_ID fd, mesh.faces()) {
    int fj = 0;
    BOOST_FOREACH(F_ID fd_new, CGAL::faces_around_face(mesh.halfedge(fd), mesh)) {
      f_affinity[3*n_f+fj] = fd_new;
      ++fj;
    }
    ++n_f;
  }
}
 

void border_vertex(float* vertex, int num_vertices, int* face, int num_faces, int* v_idx) {
  Mesh mesh;
  for (int i = 0; i < num_vertices; ++i) {
    mesh.add_vertex(Point_3(vertex[3*i], vertex[3*i+1], vertex[3*i+2]));
  }
  for (int i = 0; i < num_faces; ++i) {
    mesh.add_face(V_ID(face[3*i]), V_ID(face[3*i+1]), V_ID(face[3*i+2]));
  }

 
  std::vector<edge_descriptor> border;
  PMP::border_halfedges(faces(mesh),
       mesh,
       boost::make_function_output_iterator(halfedge2edge(mesh, border)));

  for (int i = 0; i < border.size(); ++i) {
      v_idx[(size_type) target(border[i], mesh)] = 1;
      v_idx[(size_type) source(border[i], mesh)] = 1;
  }
}

void isotropic_remeshing(float* vertex, int& num_vertices, int* face, int& num_faces, double target_edge_length, int nb_iter, float* new_vertex, int* new_face, int prepared_vertices, int prepared_faces) {
  Mesh mesh;
  for (int i = 0; i < num_vertices; ++i) {
    mesh.add_vertex(Point_3(vertex[3*i], vertex[3*i+1], vertex[3*i+2]));
  }
  for (int i = 0; i < num_faces; ++i) {
    mesh.add_face(V_ID(face[3*i]), V_ID(face[3*i+1]), V_ID(face[3*i+2]));
  }

  std::vector<edge_descriptor> border;
  PMP::border_halfedges(faces(mesh),
       mesh,
       boost::make_function_output_iterator(halfedge2edge(mesh, border)));
  PMP::split_long_edges(border, target_edge_length, mesh);

  
  PMP::isotropic_remeshing(
      faces(mesh),
      target_edge_length,
      mesh,
      PMP::parameters::number_of_iterations(nb_iter)
      .protect_constraints(true)//i.e. protect border, here
  );
  num_vertices = mesh.number_of_vertices();   
  num_faces = mesh.number_of_faces();

  if (num_vertices > prepared_vertices || num_faces > prepared_faces)
    return;

  std::vector<int> reindex;
  reindex.resize(mesh.num_vertices());

  int n = 0;
  BOOST_FOREACH(V_ID vd, mesh.vertices()) {
    auto p = mesh.point(vd);
    new_vertex[3*n] = p.x();
    new_vertex[3*n+1] = p.y();
    new_vertex[3*n+2] = p.z();
    reindex[vd] = n++;
  }
  int n_f = 0;
  BOOST_FOREACH(F_ID fd, mesh.faces()) {
    int fj = 0;
    BOOST_FOREACH(V_ID vd, CGAL::vertices_around_face(mesh.halfedge(fd), mesh)) {
      new_face[3*n_f+fj] = reindex[vd];
      ++fj;
    }
    ++n_f;
  }
}


int find_convex_hull(float* vertex, int num_vertices, float* hull_location) {
  Points points, result;
  for (int i = 0; i < num_vertices; ++i) {
    points.push_back(Point_2(vertex[3*i], vertex[3*i+1]));
  }
  CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(result));
  int result_num = result.size();
  for (int i = 0; i < result_num; ++i) {
    hull_location[2*i] = result[i].x();
    hull_location[2*i + 1] = result[i].y();
  }
  return result_num;
}

