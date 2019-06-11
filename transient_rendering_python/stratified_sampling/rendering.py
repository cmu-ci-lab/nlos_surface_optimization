import numpy as np
import math
import time
from itertools import compress
from multiprocessing import Pool

from functools import partial

from scipy.spatial import Delaunay
import pyigl as igl


class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  vn = igl.eigen.MatrixXd()
  doublearea = igl.eigen.MatrixXd()

def new_render_all_collocate(mesh,opt,barycoord):
    measurement_num = opt.lighting.shape[0] 
    mesh_transient = np.zeros((measurement_num, opt.max_distance_bin))

    fn = np.array(mesh.fn)
    fid = np.array(barycoord.col(0).replicate(measurement_num,1))
    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord).replicate(measurement_num,1)
    source = igl.eigen.MatrixXd(np.reshape(np.tile(opt.lighting, (1, opt.sample_num)),(opt.sample_num*measurement_num,3)))

    
    v1 = source - intersection_p 
    d1 = np.array(v1.rowwiseNorm())
    v1 = v1.rowwiseNormalized()

    barycoord1 = igl.eigen.MatrixXd()
    barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
    fid1 = np.array(barycoord1.col(0))
    idx = np.hstack((fid == fid1))
    f_idx = np.hstack(fid[idx]).astype(int)
   
    normalMap = np.zeros((opt.sample_num*measurement_num,3)) 
    if opt.normal == 'fn':
        normalMap[idx,:] = fn[f_idx,:]
    else:
        barycoord1 = np.array(barycoord1)
        f = np.array(mesh.f)
        f = f[np.hstack(fid[idx]).astype(int),:]
        normalMap[idx,:] = np.vstack((1-barycoord1[idx,1]-barycoord1[idx,2]))*mesh.vn[f[:,0],:] + np.vstack(barycoord1[idx,1])*mesh.vn[f[:,1],:] + np.vstack(barycoord1[idx,2])*mesh.vn[f[:,2],:]

    cos_theta1 = np.zeros(opt.sample_num*measurement_num)
    cos_theta1[idx] = np.einsum('ij,ij->i', normalMap[idx,:], np.array(v1)[idx,:])
    cos_theta1[cos_theta1<0] = 0
    
    d1_new = np.squeeze(d1[idx])
    distance_bin = np.ones(opt.sample_num*measurement_num)*(opt.distance_resolution)
    distance_bin[idx] = np.ceil((d1_new *2)/opt.distance_resolution) -1

    idx = distance_bin < opt.max_distance_bin
    intensity = np.zeros(opt.sample_num*measurement_num) 
    d1_new = np.squeeze(d1[idx])
    intensity[idx] = np.divide(cos_theta1[idx]**2, (d1_new**4))
    intensity = np.reshape(intensity, (measurement_num, opt.sample_num))    

    u = np.unique(distance_bin[idx])
    for x in u:
        d_ind = np.reshape(distance_bin == x, (measurement_num, opt.sample_num))
        for m in range(measurement_num):
            mesh_transient[m, x.astype(int)] += sum(intensity[m, d_ind[m,:]])

    mesh_transient *= mesh.total_area
    mesh_transient /= opt.sample_num
    return mesh_transient


def render_all_fun(i, v, f, vn, opt):
  mesh = MESH()
  mesh.v = igl.eigen.MatrixXd(v)
  mesh.f = igl.eigen.MatrixXi(f)
  mesh.vn = vn
  igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
  igl.doublearea(mesh.v, mesh.f, mesh.doublearea)
  if opt.method == 'n':
    transient = mesh_sampling_collocate(mesh, opt.lighting[int(i),:], opt.lighting_normal[int(i),:], opt)
  else:
    transient = stratified_mesh_sampling_collocate(mesh, opt.lighting[int(i),:], opt.lighting_normal[int(i),:], opt)
  return transient

def render_all_collocate(mesh, opt, measurement_num):
  if opt.thread_num == 1:
    transient = np.zeros((measurement_num, opt.max_distance_bin))
    for i in range(measurement_num):
        transient[i] = mesh_sampling_collocate(mesh, opt.lighting[i,:], opt.lighting_normal[i,:], opt)
  else:
    v = np.array(mesh.v)
    f = np.array(mesh.f)
    fn = np.array(mesh.fn)
    vn = np.array(mesh.vn)
    myfunc = partial(render_all_fun, v=v, f=f, vn=vn,opt=opt)
    transient = np.zeros((measurement_num, opt.max_distance_bin))
    for start in range(opt.thread_batch):
      index = np.ones(measurement_num)*0  
      with Pool(processes=opt.thread_num) as p:  
        result = p.map(myfunc, index, chunksize=1)
        p.close()
        p.join()
        transient += np.asarray(result)
    transient /= opt.thread_batch
  return transient

def stratified_random_barycoord(mesh, sample_num):
    face_num = mesh.f.rows()
    barycoord = np.empty((sample_num*face_num,3))
    barycoord[:,1] = np.random.random(sample_num*face_num)
    barycoord[:,2] = (1 - barycoord[:,1])*np.random.random(sample_num*face_num)
    barycoord[:,0] = np.tile(np.arange(face_num),(1,sample_num))[0]
    barycoord = igl.eigen.MatrixXd(barycoord)
    return barycoord

def random_barycoord(mesh, sample_num):
    barycoord = np.empty((sample_num,3))
    barycoord[:,1] = np.random.random(sample_num)
    barycoord[:,2] = (1 - barycoord[:,1])*np.random.random(sample_num)
    
    face_area = np.array(mesh.doublearea)/2
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

    barycoord = igl.eigen.MatrixXd(barycoord)
    return barycoord

def stratified_mesh_sampling_collocate(mesh, lighting, lighting_normal, opt):
    face_num = mesh.f.rows()
    barycoord = stratified_random_barycoord(mesh, opt.sample_num)
    mesh_transient = np.zeros(opt.max_distance_bin)
    fn = np.array(mesh.fn)
    fid = np.array(barycoord.col(0))
    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)

    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num*face_num,1)))

    v1 = source - intersection_p 
    d1 = np.array(v1.rowwiseNorm())
    v1 = v1.rowwiseNormalized()

    barycoord1 = igl.eigen.MatrixXd()
    barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
    fid1 = np.array(barycoord1.col(0))

    idx = np.hstack((fid == fid1))
        
    normalMap = np.zeros((opt.sample_num*face_num,3)) 
    if opt.normal == 'fn':
        normalMap[idx,:] = fn[np.hstack(fid[idx]).astype(int),:]
    else:
        barycoord1 = np.array(barycoord1)
        f = np.array(mesh.f)
        f = f[np.hstack(fid[idx]).astype(int),:]
        normalMap[idx,:] = np.vstack((1-barycoord1[idx,1]-barycoord1[idx,2]))*mesh.vn[f[:,0],:] + np.vstack(barycoord1[idx,1])*mesh.vn[f[:,1],:] + np.vstack(barycoord1[idx,2])*mesh.vn[f[:,2],:]

    cos_theta1 = np.zeros(opt.sample_num*face_num)
    cos_theta1[idx] = np.einsum('ij,ij->i', normalMap[idx,:], np.array(v1)[idx,:])
    cos_theta1[cos_theta1<0] = 0
    
    d1_new = np.squeeze(d1[idx])
    distance_bin = np.ones(opt.sample_num*face_num)*(opt.distance_resolution)
    distance_bin[idx] = np.ceil((d1_new *2)/opt.distance_resolution) -1

    idx = distance_bin < opt.max_distance_bin
    intensity = np.zeros(opt.sample_num*face_num) 
    d1_new = np.squeeze(d1[idx])
    intensity[idx] = np.divide(cos_theta1[idx]**2, (d1_new**4))
    
    face_area = np.array(mesh.doublearea)/2
    intensity *= face_area[fid.astype(int)].flatten()

    u = np.unique(distance_bin[idx])
    for x in u:
        mesh_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    mesh_transient /= opt.sample_num
    return mesh_transient


def old_stratified_mesh_sampling_collocate(mesh, lighting, lighting_normal, opt):
    face_area = np.array(mesh.doublearea)/2
    mesh.total_area = sum(face_area)
    mesh_transient = np.zeros(opt.max_distance_bin)
    fn = np.array(mesh.fn)

    for face in range(fn.shape[0]):
        barycoord = np.ones((opt.sample_num,3))*face
        barycoord[:,1] = np.random.random(opt.sample_num)
        barycoord[:,2] = (1 - barycoord[:,1])*np.random.random(opt.sample_num)
        barycoord = igl.eigen.MatrixXd(barycoord)   

        intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)

        source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num,1)))

        v1 = source - intersection_p 
        d1 = np.array(v1.rowwiseNorm())
        v1 = v1.rowwiseNormalized()

        barycoord1 = igl.eigen.MatrixXd()
        barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
        fid1 = np.array(barycoord1.col(0))

        idx = np.hstack((fid1 == face))
        if not idx.any():
            continue
   
        normalMap = np.zeros((opt.sample_num,3)) 
        if opt.normal == 'fn':
            normalMap[idx,:] = fn[np.hstack(fid1[idx]).astype(int),:]
        else:
            barycoord1 = np.array(barycoord1)
            f = np.array(mesh.f)
            f = f[np.hstack(fid1[idx]).astype(int),:]
            normalMap[idx,:] = np.vstack((1-barycoord1[idx,1]-barycoord1[idx,2]))*mesh.vn[f[:,0],:] + np.vstack(barycoord1[idx,1])*mesh.vn[f[:,1],:] + np.vstack(barycoord1[idx,2])*mesh.vn[f[:,2],:]

        cos_theta1 = np.zeros(opt.sample_num)
        cos_theta1[idx] = np.einsum('ij,ij->i', normalMap[idx,:], np.array(v1)[idx,:])
        cos_theta1[cos_theta1<0] = 0
    
        d1_new = np.squeeze(d1[idx])
        distance_bin = np.ones(opt.sample_num)*(opt.distance_resolution)
        distance_bin[idx] = np.ceil((d1_new *2)/opt.distance_resolution) -1

        idx = distance_bin < opt.max_distance_bin
        intensity = np.zeros(opt.sample_num) 
        d1_new = np.squeeze(d1[idx])
        intensity[idx] = np.divide(cos_theta1[idx]**2, (d1_new**4))

        u = np.unique(distance_bin[idx])
        for x in u:
            mesh_transient[x.astype(int)] += (sum(intensity[distance_bin == x]))*face_area[face]
            #mesh_transient[x.astype(int)] += (sum(intensity[distance_bin == x]))
    #mesh_transient *= face_area.shape[0]
    #mesh_transient *= mesh.total_area
    mesh_transient /= opt.sample_num
    return mesh_transient

def mesh_sampling_collocate(mesh, lighting, lighting_normal, opt):
    barycoord = random_barycoord(mesh, opt.sample_num)
    mesh_transient = np.zeros(opt.max_distance_bin)
    fn = np.array(mesh.fn)
    fid = np.array(barycoord.col(0))
    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)

    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num,1)))

    v1 = source - intersection_p 
    d1 = np.array(v1.rowwiseNorm())
    v1 = v1.rowwiseNormalized()

    barycoord1 = igl.eigen.MatrixXd()
    barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
    fid1 = np.array(barycoord1.col(0))

    idx = np.hstack((fid == fid1))
   
    normalMap = np.zeros((opt.sample_num,3)) 
    if opt.normal == 'fn':
        normalMap[idx,:] = fn[np.hstack(fid[idx]).astype(int),:]
    else:
        barycoord1 = np.array(barycoord1)
        f = np.array(mesh.f)
        f = f[np.hstack(fid[idx]).astype(int),:]
        normalMap[idx,:] = np.vstack((1-barycoord1[idx,1]-barycoord1[idx,2]))*mesh.vn[f[:,0],:] + np.vstack(barycoord1[idx,1])*mesh.vn[f[:,1],:] + np.vstack(barycoord1[idx,2])*mesh.vn[f[:,2],:]

    cos_theta1 = np.zeros(opt.sample_num)
    cos_theta1[idx] = np.einsum('ij,ij->i', normalMap[idx,:], np.array(v1)[idx,:])
    cos_theta1[cos_theta1<0] = 0
    
    d1_new = np.squeeze(d1[idx])
    distance_bin = np.ones(opt.sample_num)*(opt.distance_resolution)
    distance_bin[idx] = np.ceil((d1_new *2)/opt.distance_resolution) -1

    idx = distance_bin < opt.max_distance_bin
    intensity = np.zeros(opt.sample_num) 
    d1_new = np.squeeze(d1[idx])
    intensity[idx] = np.divide(cos_theta1[idx]**2, (d1_new**4))

    u = np.unique(distance_bin[idx])
    for x in u:
        mesh_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    mesh_transient *= mesh.total_area
    mesh_transient /= opt.sample_num
    return mesh_transient

def mesh_sampling(mesh, barycoord, lighting, sensor, lighting_normal, sensor_normal, opt):
    mesh_transient = np.zeros(opt.max_distance_bin)
    
    fn = np.array(mesh.fn)
    fid = np.array(barycoord.col(0))
    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)

    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num,1)))

    v1 = source - intersection_p 
    d1 = np.array(v1.rowwiseNorm())
    v1 = v1.rowwiseNormalized()

    barycoord1 = igl.eigen.MatrixXd()
    barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
    fid1 = np.array(barycoord1.col(0))

    idx = list(compress(range(opt.sample_num), fid == fid1))

    source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx),1)))
    intersection_p_subset = igl.eigen.MatrixXd(np.array(intersection_p)[idx,:])
    v2 = source - intersection_p_subset
    d2 = np.array(v2.rowwiseNorm())
    v2 = v2.rowwiseNormalized()

    barycoord2 = igl.eigen.MatrixXd()
    barycoord2 = igl.embree.line_mesh_intersection(source, -v2, mesh.v, mesh.f)
    fid2 = np.array(barycoord2.col(0))
    idx1 = list(compress(idx, fid1[idx] == fid2))
    if len(idx1) == 0:
        return mesh_transient
    idx2 = list(compress(range(len(fid2)), fid1[idx] == fid2))
    if opt.normal == 'fn':
        normalMap = mesh.fn[np.hstack(fid2[idx2]).astype(int),:]
    else:
        barycoord2 = np.array(barycoord2)
        barycoord2 = barycoord2[idx2,:]
        f = np.array(mesh.f)
        f = f[np.hstack(fid2[idx2]).astype(int),:]
        normalMap = np.vstack((1-barycoord2[:,1]-barycoord2[:,2]))*mesh.vn[f[:,0],:] + np.vstack(barycoord2[:,1])*mesh.vn[f[:,1],:] + np.vstack(barycoord2[:,2])*mesh.vn[f[:,2],:]

    cos_theta1 = np.einsum('ij,ij->i', normalMap, np.array(v1)[idx1,:])
    cos_theta1[cos_theta1<0] = 0
    cos_theta2 = np.einsum('ij,ij->i', normalMap, np.array(v2)[idx2,:])
    cos_theta2[cos_theta2<0] = 0
    d1_new = np.squeeze(d1[idx1])
    d2_new = np.squeeze(d2[idx2])
    distance_bin = np.ceil((d1_new + d2_new)/opt.distance_resolution) -1

    idx = list(compress(range(len(distance_bin)), distance_bin < opt.max_distance_bin))
    intensity = np.divide(cos_theta1*cos_theta2, (d1_new**2)*(d2_new**2))

    u = np.unique(distance_bin[idx])
    for x in u:
        mesh_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    mesh_transient *= mesh.total_area
    mesh_transient /= opt.sample_num
    return mesh_transient
