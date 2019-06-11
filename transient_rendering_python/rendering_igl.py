import numpy as np
import math
from itertools import compress

import element_wise_manipulation
import pyigl as igl

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

def mesh_grad_sampling(mesh, barycoord, lighting, sensor, lighting_normal, sensor_normal, opt):

    face_area = np.array(mesh.doublearea)/2
    grad = np.zeros((opt.max_distance_bin, 3*mesh.v.rows()))

    for vertex in range(mesh.v.rows()):

        related_face = list(compress(range(mesh.f.rows()), np.sum(np.array(mesh.f) == vertex, 1) > 0 ))

        for f in related_face:
            face = np.array(mesh.f.row(f))[0]
            vertex_idx = list(compress(range(3), face == vertex))
            vertex_idx = vertex_idx[0]
            ind = np.squeeze(np.array(barycoord.col(0)) == f) 
            
            u = np.array(barycoord.col(1))[ind,:]
            v = np.array(barycoord.col(2))[ind,:]
            phi = 1-u-v

            sample_num = np.sum(ind)
            barycoord_subset = igl.eigen.MatrixXd(np.array(barycoord)[ind,:])
            sample_point = igl.barycentric_to_global(mesh.v, mesh.f, barycoord_subset)

            source = igl.eigen.MatrixXd(np.tile(lighting, (sample_num, 1)))
            v1 = source - sample_point
            d1 = np.array(v1.rowwiseNorm())
            v1 = v1.rowwiseNormalized()
            
            barycoord1 = igl.eigen.MatrixXd()
            barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)
            fid1 = np.array(barycoord1.col(0))

            idx = list(compress(range(sample_num), fid1 == f))

            source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx),1)))

            sample_point_subset = igl.eigen.MatrixXd(np.array(sample_point)[idx,:])
            v2 = source - sample_point_subset
            d2 = np.array(v2.rowwiseNorm())
            v2 = v2.rowwiseNormalized()

            barycoord2 = igl.eigen.MatrixXd()
            barycoord2 = igl.embree.line_mesh_intersection(source, -v2, mesh.v, mesh.f)
            fid2 = np.array(barycoord2.col(0))
            idx1 = list(compress(idx, fid2 == f))
            idx2 = list(compress(range(len(fid2)), fid2 == f))
            normalMap = mesh.fn[f,:]

            u_new = u[idx1]
            v_new = v[idx1]
            phi_new = phi[idx1]
            
            v1_new = np.array(v1)[idx1,:]
            v2_new = np.array(v2)[idx2,:]
            cos_theta1 = np.dot(v1_new, normalMap)
            cos_theta1[cos_theta1<0] = 0
            cos_theta2 = np.dot(v2_new, normalMap)
            cos_theta2[cos_theta2<0] = 0
            d1_new = np.squeeze(d1[idx1])
            d2_new = np.squeeze(d2[idx2])
            distance_bin = np.ceil((d1_new + d2_new)/opt.distance_resolution) -1
            idx = range(len(distance_bin))
            #idx = list(compress(range(len(distance_bin)), distance_bin < opt.max_distance_bin))
            intensity = np.divide(cos_theta1*cos_theta2, (d1_new**2)*(d2_new**2))

            gx1_tmp = d2_new[idx]*cos_theta2[idx] + d1_new[idx]*cos_theta1[idx]
            gx1 = normalMap*np.vstack(gx1_tmp)
            
            gx2_tmp1 = v1_new[idx,:]*np.vstack(d2_new[idx]) + v2_new[idx,:]*np.vstack(d1_new[idx])
            gx2_tmp2 = cos_theta1[idx] * cos_theta2[idx]
            gx2 = gx2_tmp1*np.vstack(gx2_tmp2)
            
            t1_tmp1 = -gx1+3*gx2
            t1_tmp2 = (d1_new[idx]**3)*(d2_new[idx]**3)
            t1 = t1_tmp1/np.vstack(t1_tmp2)
            
            t2 = normalMap*np.vstack(intensity[idx]) 

            gn_tmp1 = v1_new[idx,:]*np.vstack(cos_theta2[idx]) + v2_new[idx,:]*np.vstack(cos_theta1[idx])
            gn_tmp2 = (d1_new[idx]**2)*(d2_new[idx]**2)
            gn_tmp = gn_tmp1/np.vstack(gn_tmp2)
            cos_tmp = np.dot(gn_tmp, normalMap)
            gn = gn_tmp - normalMap*np.vstack(cos_tmp)
            
            t2 = (t2 + gn)/(2*face_area[f])
            if vertex_idx == 0:
                e = np.array(mesh.v.row(face[2]) - mesh.v.row(face[1]))
                g = t1*phi_new[idx] + np.cross(t2, e)
            elif vertex_idx == 1:
                e = np.array(mesh.v.row(face[0]) - mesh.v.row(face[2]))
                g = t1*u_new[idx] + np.cross(t2, e)
            else: 
                e = np.array(mesh.v.row(face[1]) - mesh.v.row(face[0]))
                g = t1*v_new[idx] + np.cross(t2, e)


            distance_new = distance_bin[idx]
            unique_bin = np.unique(distance_new)
            for x in unique_bin:
                grad[x.astype(int),3*vertex:3*(vertex+1)] += np.sum(g[distance_new == x,:],0)*face_area[f]

    grad /= opt.sample_num
    return grad

def mesh_sampling(mesh, barycoord, lighting, sensor, lighting_normal, sensor_normal, opt):
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

    mesh_transient = np.zeros(opt.max_distance_bin)
    u = np.unique(distance_bin[idx])
    for x in u:
        mesh_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    mesh_transient *= mesh.total_area
    mesh_transient /= opt.sample_num
    return mesh_transient


def angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt):
    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num, 1)))
    n = igl.eigen.MatrixXd(direction)


    angular_transient = np.zeros(opt.max_distance_bin)
    barycoord = igl.eigen.MatrixXd()
    barycoord = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)

    fid = np.array(barycoord.col(0)).astype(int)
    idx = list(compress(range(opt.sample_num), fid != -1))


    normalMap = np.empty((opt.sample_num,3))
    normalMap[:] = np.nan
    v11 = np.empty(opt.sample_num)
    v11[:] = np.nan
    d1 = np.empty(opt.sample_num)
    d1[:] = np.nan

    v2 = np.empty((opt.sample_num, 3))
    v2[:] = np.nan
    d2 = np.empty(opt.sample_num)
    d2[:] = np.nan
    cos_theta2 = np.empty(opt.sample_num)
    cos_theta2[:] = np.nan
    distance_bin = np.empty(opt.sample_num)
    distance_bin[:] = opt.max_distance_bin + 1
    intensity = np.empty(opt.sample_num)
    intensity[:] = np.nan



    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)
    intersection_p = np.array(intersection_p)
    normalMap[idx,:] = mesh.fn[np.hstack(fid[idx]),:]


    v11[idx] = lighting[0] - intersection_p[idx,0]
    d1[idx] = np.abs(np.divide(v11[idx], direction[idx,0]))

    v2[idx,:] = sensor - intersection_p[idx,:]
    d2[idx] = np.sqrt(np.sum(v2[idx,:]**2, axis = 1)) 
    v2[idx,:] = element_wise_manipulation.element_divide2(v2[idx,:], d2[idx])

    source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx), 1)))
    n = igl.eigen.MatrixXd(-v2[idx,:])

    barycoord2 = igl.eigen.MatrixXd()
    barycoord2 = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)
    fid2 = np.array(barycoord2.col(0))

    idx = list(compress(idx, fid[idx] == fid2))

    cos_theta2[idx] = np.einsum('ij,ij->i', normalMap[idx,:], v2[idx,:])
    less_than_zero = list(compress(idx, cos_theta2[idx] < 0))
    if len(less_than_zero) != 0:
        cos_theta2[less_than_zero] = 0
    

    distance_bin[idx] = np.ceil((d1[idx]+d2[idx])/opt.distance_resolution) -1

    inds = list(compress(range(opt.sample_num), distance_bin<=opt.max_distance_bin))
    intensity[inds] = np.divide(cos_theta2[inds], d2[inds]**2)

    u = np.unique(distance_bin[inds])

    for x in u:
        angular_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num
    return angular_transient
