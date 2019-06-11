import numpy as np
import math

from itertools import compress
import torch
from torch.autograd import Variable

import mesh_intersection_grad_igl
import mesh_intersection_grad



def initialize_variable(sz, val):
   d1 = np.empty(sz)
   d1[:] = val
   return Variable(torch.from_numpy(d1).cuda())
  

def angular_sampling(mesh, direction, triangleIndexMap, lighting, sensor, lighting_normal, sensor_normal, opt):
    angular_transient = Variable(torch.DoubleTensor(opt.max_distance_bin).fill_(0).cuda())

    triangleIndexMap = Variable(torch.from_numpy(triangleIndexMap).long().cuda())
    direction = Variable(torch.from_numpy(direction).cuda())
    lighting = Variable(torch.from_numpy(lighting).cuda())
    sensor = Variable(torch.from_numpy(sensor).cuda()) 

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

    distance_bin = Variable(torch.LongTensor(opt.sample_num).fill_(opt.max_distance_bin+1).cuda())

    inds = Variable(torch.squeeze((triangleIndexMap > -1).data.nonzero()))

    for i in inds:
        uMap[i], vMap[i], d1[i] = mesh_intersection_grad_igl.intersect_ray_mesh_one_direction(mesh, torch.squeeze(direction[i,:]), torch.squeeze(mesh.f[triangleIndexMap[i],:]), lighting)


    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)
    data = mesh_intersection_grad.element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:]) + mesh_intersection_grad.element_multiply2(uMap[inds], mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:]) + mesh_intersection_grad.element_multiply2(vMap[inds], mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:])
    intersection_p.index_copy_(0, inds, data)

    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = torch.sqrt(torch.sum(v2[inds,:]**2, 1))
    v2[inds,:] = mesh_intersection_grad.element_divide2(v2[inds,:], d2[inds])

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
