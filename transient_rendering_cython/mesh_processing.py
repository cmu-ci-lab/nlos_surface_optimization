import sys
import numpy as np
import scipy as sc
import math

def barycentric_to_global(mesh_v, mesh_f, barycoord):
    p = np.empty(barycoord.shape, dtype=np.float32)
    fid = barycoord[:,0]
    idx = (fid >= 0)
    f = mesh_f[np.hstack(fid[idx]).astype(int),:]
    p[idx,:] = np.vstack((1-barycoord[idx,1]-barycoord[idx,2]))*mesh_v[f[:,0],:] + np.vstack(barycoord[idx,1])*mesh_v[f[:,1],:] + np.vstack(barycoord[idx,2])*mesh_v[f[:,2],:]
    return p

def face_normal_and_area(v,f):
  p1 = v[f[:,0], :]
  p2 = v[f[:,1], :]
  p3 = v[f[:,2], :]
  n = np.cross(p2 - p1, p3 - p1, axis=1)
  n += sys.float_info.epsilon
  d = np.linalg.norm(n, axis=1)
  n = np.divide(n, np.vstack(d))
  return n, d/2 


def cotmatrix(V,F):
    i1 = F[:,0]
    i2 = F[:,1]
    i3 = F[:,2]
    
    v1 = V[i3,:] - V[i2,:]
    v2 = V[i1,:] - V[i3,:]
    v3 = V[i2,:] - V[i1,:]

    n = np.cross(v1,v2,axis=1)
    dblA = np.linalg.norm(n, axis=1)
    A = dblA/2

    cot12 = -einsum('ij,ij->i', v1, v2)/A
    cot22 = -einsum('ij,ij->i', v2, v3)/A
    cot31 = -einsum('ij,ij->i', v3, v1)/A

    diag1 = -cot12 - cot32
    diag2 = -cot12 - cot23
    diag3 = -cot31 - cot23

    i = np.hstack((i1,i2,i2,i3,i3,i1,i1,i2,i3))
    j = np.hstack((i2,i1,i3,i2,i1,i3,i1,i2,ie))
    v = np.hstack((cot12, cot12, cot23, cot23, cot31, cot31, diag1, diag2, diag3))
    L = scipy.sparse.coo_matrix((v,(i,j)), shape=(V.shape[0],V.shape[0])).toarray()
    return L

def massmatrix(V,F):
    i1 = F[:,0]
    i2 = F[:,1]
    i3 = F[:,2]
    
    v1 = V[i3,:] - V[i2,:]
    v2 = V[i1,:] - V[i3,:]
    v3 = V[i2,:] - V[i1,:]

    n = np.cross(v1,v2,axis=1)
    dblA = np.linalg.norm(n, axis=1)
    
    i = np.hstack((i1,i2,i3))
    j = i
    diag_v = dblA/6
    v = np.hstack((diag_v,diag_v,diag_v))
    M = scipy.sparse.coo_matrix((v,(i,j)), shape=(V.shape[0],V.shape[0])).toarray()
    return M

def biharmonic_embedding(V,F,dim):
    L = cotmatrix(V,F) 
    M = massmatrix(V,F,'barycentric')
    ED,EV = sc.linalg.eig(L,M)
    idx = np.argsort(ED)
    ED = ED[idx]
    EV = EV[:,idx] 
    EV = EV[:,1:dim+1]
    ED = ED[1:dim+1]
    B = EV@np.diag(1/abs(ED))
    return B


def blue_noise_random_barycoord(mesh, sample_num):
    face_num = mesh.f.shape[0]
    k = 10
    BV = biharmonic_embedding(mesh.v, mesh.f, k)
    #TODO

    barycoord = np.empty((sample_num*face_num,3))
    barycoord[:,1] = np.random.random(sample_num*face_num)
    barycoord[:,2] = (1 - barycoord[:,1])*np.random.random(sample_num*face_num)
    barycoord[:,0] = np.tile(np.arange(face_num),(1,sample_num))[0]
    barycoord = np.array(barycoord, dtype=np.float32)
    return barycoord

def stratified_random_barycoord(mesh, sample_num_total):
    face_num = mesh.f.shape[0]
    sample_num = math.floor((sample_num_total-1)/face_num) + 1
    barycoord = np.empty((sample_num*face_num,3))
    sqrt_t = np.sqrt(np.random.random(sample_num*face_num))
    s = np.random.random(sample_num*face_num)

    barycoord[:,1] = 1-sqrt_t
    barycoord[:,2] = np.multiply(s,sqrt_t)
    barycoord[:,0] = np.tile(np.arange(face_num),(1,sample_num))[0]
    barycoord = np.array(barycoord, dtype=np.float32)
    return barycoord


def random_barycoord(mesh, sample_num):
    barycoord = np.empty((sample_num,3))
    sqrt_t = np.sqrt(np.random.random(sample_num))
    s = np.random.random(sample_num)

    barycoord[:,1] = 1-sqrt_t
    barycoord[:,2] = np.multiply(s,sqrt_t)
    
    face_area = mesh.face_area
    mesh.total_area = sum(face_area)
    face_area_ratio = face_area/mesh.total_area
    face_area_ratio = np.reshape(face_area_ratio, face_area_ratio.size)

    face_trial = np.random.multinomial(sample_num, face_area_ratio)
    count = 0

    for x in range(face_trial.size):
        if face_trial[x] == 0:
            continue
        barycoord[count:count+face_trial[x],0] = x
        count += face_trial[x]
    barycoord = np.array(barycoord, dtype = np.float32)
    return barycoord
