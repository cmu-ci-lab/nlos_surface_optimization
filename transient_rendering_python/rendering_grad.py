import numpy as np
import math

from itertools import compress
import torch
from torch.autograd import Variable

import mesh_intersection_grad

def initialize_variable(sz, val):
   d1 = np.empty(sz)
   d1[:] = val
   return Variable(torch.from_numpy(d1))
  

def angular_sampling(mesh, direction, lighting, sensor, lighting_normal, sensor_normal, opt):
    angular_transient = Variable(torch.DoubleTensor(opt.max_distance_bin).fill_(0))
 
    direction = Variable(torch.from_numpy(direction))
    lighting = Variable(torch.from_numpy(lighting))
    sensor = Variable(torch.from_numpy(sensor)) 
    intersect, t, u, v = mesh_intersection_grad.intersect_ray_mesh_batch_directions(lighting, direction, mesh, opt.epsilon)
     

    d1 = initialize_variable(opt.sample_num, np.nan)
    d2 = initialize_variable(opt.sample_num, np.nan)
    intensity = initialize_variable(opt.sample_num, np.nan)
    uMap = initialize_variable(opt.sample_num, np.nan)
    vMap = initialize_variable(opt.sample_num, np.nan)
    cos_theta2 = initialize_variable(opt.sample_num, 0)
    intersection_p = initialize_variable((opt.sample_num,3), np.nan)
    normalMap = initialize_variable((opt.sample_num,3), np.nan)
    v2 = initialize_variable((opt.sample_num,3), np.nan)
    tmp_e1 = initialize_variable((opt.sample_num,3), np.nan)
    tmp_e2 = initialize_variable((opt.sample_num,3), np.nan)
    fn = initialize_variable((opt.sample_num,3), np.nan)
    fn_len = initialize_variable(opt.sample_num, np.nan)

    distance_bin = Variable(torch.LongTensor(opt.sample_num).fill_(opt.max_distance_bin+1))
    triangleIndexMap = Variable(torch.LongTensor(opt.sample_num).fill_(-1))
    
    face_num = mesh.f.shape[0]
    
    for i in range(opt.sample_num):

        intersectInds = Variable(torch.squeeze(intersect[:,i].data.nonzero()))
        #intersectInds = intersect[:,i].data.nonzero()
        intersection_num = len(intersectInds)
        if intersection_num == 0:
            continue 
        elif intersection_num == 1:
            d1[i] = torch.abs(torch.index_select(t[:,i], 0,  intersectInds))            
            triangleIndex = intersectInds
        else:

            val, nearestIntersection = torch.min(torch.abs(torch.index_select(t[:,i],0,intersectInds)),0)
            d1[i] = val
            triangleIndex = intersectInds[nearestIntersection]

        uMap[i] = torch.index_select(u[:,i], 0, triangleIndex)
        vMap[i] = torch.index_select(v[:,i], 0, triangleIndex)
        triangleIndexMap[i] = triangleIndex

    inds = Variable(torch.squeeze((triangleIndexMap > -1).data.nonzero()))
  
   
    if len(inds) == 0:
        print('no intersection 1')
        return angular_transient

    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)

    data = mesh_intersection_grad.element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:]) + mesh_intersection_grad.element_multiply2(uMap[inds], mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:]) + mesh_intersection_grad.element_multiply2(vMap[inds], mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:])
    intersection_p.index_copy_(0, inds, data)


    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = torch.sqrt(torch.sum(v2[inds,:]**2, 1))
    v2[inds,:] = mesh_intersection_grad.element_divide2(v2[inds,:], d2[inds])
    intersect, t, _, _ = mesh_intersection_grad.intersect_ray_mesh_batch_directions(sensor, -v2[inds,:], mesh, opt.epsilon)
  
    for i in range(len(inds)):
        index = Variable(torch.squeeze(((t[:,i] <= 0).__or__(t[:,i] > d2[inds[i]]+opt.epsilon)).data.nonzero()))
        if len(index) > 0:
            intersect[:,i] = intersect[:,i].index_fill(0, index, 0)
    
    inds_tmp =   Variable(torch.squeeze((torch.sum(intersect,0) <= 1).data.nonzero()))
    
    if len(inds_tmp) == 0:
        print('no intersection 2')
        return angular_transient
    
    inds = torch.index_select(inds, 0, inds_tmp)
   
     
    
    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)
  
    
    tmp_e1[inds,:] = mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 
    tmp_e2[inds,:] = mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 
   
    fn[inds,:] = torch.cross(tmp_e1[inds,:], tmp_e2[inds,:], 1) 
    fn_len[inds] = torch.sqrt(torch.sum(fn[inds,:]**2,1)) 
    normalMap[inds,:] = mesh_intersection_grad.element_divide2(fn[inds,:], fn_len[inds])
  
    cos_theta2[inds] = torch.sum(torch.mul(normalMap[inds,:], v2[inds,:]),1)
  
   
    index = Variable(torch.squeeze((cos_theta2<0).data.nonzero())) 
    if len(index) != 0:
        cos_theta2 = cos_theta2.index_fill(0, index, 0)
    
    
    distance_bin[inds] = torch.ceil((d1[inds]+d2[inds])/opt.distance_resolution).long() -1

    inds = Variable(torch.squeeze((distance_bin<opt.max_distance_bin).data.nonzero()))
    val = torch.div(cos_theta2.index_select(0,inds), d2.index_select(0,inds)**2)
    intensity.index_copy_(0,inds,val)
   
    
    angular_transient.index_add_(0, distance_bin[inds],  intensity[inds])

    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num
    return angular_transient
