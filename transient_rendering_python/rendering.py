import numpy as np
import math


from itertools import compress
import mesh_intersection

def angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt):
    angular_transient = np.zeros(opt.max_distance_bin)
    intersect, t, u, v = mesh_intersection.intersect_ray_mesh_batch_directions(lighting, direction, mesh, opt.epsilon)


    d1 = np.empty(opt.sample_num)
    d1[:] = np.nan
    d2 = np.empty(opt.sample_num)
    d2[:] = np.nan
    distance_bin = np.empty(opt.sample_num)
    distance_bin[:] = opt.max_distance_bin + 1
    intensity = np.empty(opt.sample_num)
    intensity[:] = np.nan 
    uMap = np.empty(opt.sample_num)
    uMap[:] = np.nan
    vMap = np.empty(opt.sample_num)
    vMap[:] = np.nan
    triangleIndexMap = np.empty(opt.sample_num)
    triangleIndexMap[:] = np.nan
    cos_theta2 = np.empty(opt.sample_num)
    cos_theta2[:] = np.nan
    intersection_p = np.empty((opt.sample_num,3))
    intersection_p[:] = np.nan
    normalMap = np.empty((opt.sample_num,3))
    normalMap[:] = np.nan
    v2 = np.empty((opt.sample_num,3))
    v2[:] = np.nan

    face_num = mesh.f.shape[0]
    for i in range(opt.sample_num):
        intersectInds = list(compress(range(face_num), intersect[:,i]))
        intersection_num = len(intersectInds)
        if intersection_num == 0:
            continue 
        elif intersection_num == 1:
            d1[i] = abs(t[intersectInds,i])
            triangleIndex = intersectInds[0]
        else:
            nearestIntersection = np.argmin(abs(t[intersectInds,i]))
            triangleIndex = intersectInds[nearestIntersection]
            d1[i] = t[triangleIndex, i]

        uMap[i] = u[triangleIndex,i]
        vMap[i] = v[triangleIndex,i]
        triangleIndexMap[i] = triangleIndex

    inds = list(compress(range(opt.sample_num), np.logical_not(np.isnan(triangleIndexMap)) ))

    triangleIndexMap_input = triangleIndexMap[inds].astype(int)  
    intersection_p[inds,:] = mesh_intersection.element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[mesh.f[triangleIndexMap_input,2],:]) + mesh_intersection.element_multiply2(uMap[inds], mesh.v[mesh.f[triangleIndexMap_input,0],:]) + mesh_intersection.element_multiply2(vMap[inds], mesh.v[mesh.f[triangleIndexMap_input,1],:])


    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = np.sqrt(np.sum(v2[inds,:]**2, axis = 1))
    v2[inds,:] = mesh_intersection.element_divide2(v2[inds,:], d2[inds])


    intersect, t, _, _ = mesh_intersection.intersect_ray_mesh_batch_directions(sensor, -v2[inds,:], mesh, opt.epsilon)
    intersect[t<=0] = False
    intersect[t>d2[inds]+opt.epsilon] = False


    inds = list(compress([x for x in inds], np.sum(intersect, axis=0)<=1)) 
    triangleIndexMap_input = triangleIndexMap[inds].astype(int)  
    if opt.normal == 'n':
        normalMap[inds,:] = mesh_intersection.element_multiply2(1-uMap[inds] - vMap[inds], mesh.n[mesh.f[triangleIndexMap_input,2],:]) + mesh_intersection.element_multiply2(uMap[inds], mesh.n[mesh.f[triangleIndexMap_input,0],:]) + mesh_intersection.element_multiply2(vMap[inds], mesh.n[mesh.f[triangleIndexMap_input,1],:])
        normalMap[inds,:] = mesh_intersection.element_divide2(normalMap[inds,:], np.sqrt(np.sum(normalMap[inds,:]**2, axis = 1)))

    else:
        normalMap[inds,:] = mesh.fn[triangleIndexMap_input,:]


    cos_theta2[inds] = np.einsum('ij,ij->i', normalMap[inds,:], v2[inds,:])
    cos_theta2[cos_theta2 < 0] = 0
    distance_bin[inds] = np.ceil((d1[inds]+d2[inds])/opt.distance_resolution) -1

    inds = list(compress(range(opt.sample_num), distance_bin<=opt.max_distance_bin))
    intensity[inds] = np.divide(cos_theta2[inds], d2[inds]**2)

    u = np.unique(distance_bin[inds])

    for x in u:
        angular_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    angular_transient *= 2*math.pi
    return angular_transient/opt.sample_num
