import numpy as np
import scipy.io
import sys, os
from itertools import compress
sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
#sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math


import pyigl as igl


class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()


mesh_location = os.getcwd() + '/space_carving_mesh.obj'
space_carving_mesh = MESH()
read_file = igl.readOBJ(mesh_location, space_carving_mesh.v, space_carving_mesh.f)

igl.per_face_normals(space_carving_mesh.v, space_carving_mesh.f, space_carving_mesh.fn)

P = igl.eigen.MatrixXd([[0,0,0], [0,0,0.2], [0,0.1,2]])
print(P)
S = igl.eigen.MatrixXd()
I = igl.eigen.MatrixXi()
C = igl.eigen.MatrixXd()
N = igl.eigen.MatrixXd()
igl.signed_distance(P, space_carving_mesh.v, space_carving_mesh.f, igl.SignedDistanceType(0), S,I,C,N)


for x in list(compress(range(P.rows()), S>0)):
    P.setRow(x, C.row(x))

print(P) 
