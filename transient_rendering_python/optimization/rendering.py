import numpy as np
import math
from itertools import compress
import sys
import torch
import time
import threading
import rotation_matrix
import pyigl as igl

def find_triangle(mesh, direction, lighting, sensor, opt):
    sample_num = direction.shape[0]
    source = igl.eigen.MatrixXd(np.tile(lighting, (sample_num,1)))
    n = igl.eigen.MatrixXd(direction)

    barycoord = igl.eigen.MatrixXd()
    barycoord = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)
    fid = np.array(barycoord.col(0).transpose())[0]
    idx = list(compress(range(sample_num), fid != -1))

    intersection_p = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)
    intersection_p = np.array(intersection_p)
    
    v1 = np.empty((sample_num, 3))
    d1 = np.empty(sample_num)

    v1[idx,:] = lighting - intersection_p[idx,:]
    d1[idx] = np.sqrt(np.sum(v1[idx,:]**2, axis = 1)) 

    v2 = np.empty((sample_num, 3))
    d2 = np.empty(sample_num)

    v2[idx,:] = sensor - intersection_p[idx,:]
    d2[idx] = np.sqrt(np.sum(v2[idx,:]**2, axis = 1)) 
    v2[idx,:] = element_divide2_np(v2[idx,:], d2[idx])

    source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx), 1)))
    n = igl.eigen.MatrixXd(-v2[idx,:])

    barycoord2 = igl.eigen.MatrixXd()
    barycoord2 = igl.embree.line_mesh_intersection(source, n, mesh.v, mesh.f)
    fid2 = np.array(barycoord2.col(0).transpose())[0]
    
    #distance_bin = np.zeros(sample_num)
    #distance_bin[idx] = np.minimum(np.ceil((d1[idx] + d2[idx])/opt.distance_resolution) -1, opt.max_distance_bin -1 )
    

    idx = list(compress(idx, fid[idx] != fid2))
    fid[idx] = -1
    #return fid, distance_bin
    return fid

def intersect_ray_mesh_one_direction(mesh, direction, triangle, origin):
  vertex0 = torch.squeeze(mesh.v[triangle[0],:])
  vertex1 = torch.squeeze(mesh.v[triangle[1],:])
  vertex2 = torch.squeeze(mesh.v[triangle[2],:])

  edge1 = vertex0 - vertex2
  edge2 = vertex1 - vertex2

  t = edge2[2]*direction[1] - edge2[1]*direction[2]
  
  tvect1 = origin[0] - vertex2[0]
  v = t* tvect1 
  u = edge1[0]*t
  t = edge2[0] * direction[2] - edge2[2] * direction[0]
  tvect2 = origin[1] - vertex2[1]
  v = v + tvect2*t
  u = u + edge1[1]*t
  t = edge2[1]*direction[0] - edge2[0]*direction[1]
  tvect3 = origin[2] - vertex2[2]
  v = v + tvect3*t
  u = u + edge1[2]*t

  if abs(u) < sys.float_info.epsilon:
    t = 1
  else:
    t = 1/u
  u = t*v
  
  qvect1 = tvect2*edge1[2] - tvect3 * edge1[1]
  qvect2 = tvect3*edge1[0] - tvect1 * edge1[2]
  qvect3 = tvect1*edge1[1] - tvect2 * edge1[0]

  
  v = qvect1*direction[0] + qvect2*direction[1] + qvect3*direction[2]

  v = torch.mul(t,v)
  
  tmp = edge2[0]*qvect1 + edge2[1]*qvect2 + edge2[2]*qvect3
  t = torch.mul(t,tmp)
  return u, v, t

def element_multiply2(a,b):
  if a.size(0) == b.size(0):
    a = torch.transpose(a.repeat(b.size(1),1),0,1)
  else:
    a = a.repeat(b.size(0),1)
  return torch.mul(a,b)

def element_divide2(a,b):
  b = torch.transpose(b.repeat(a.size(1),1),0,1)
  return torch.div(a,b)

def element_divide2_np(a,b):
  sz = a.shape
  if b.size == sz[0]:
      b = np.tile(b, (sz[1],1)).T
  else:
      b = np.tile(b, (sz[0],1))
  return np.divide(a,b)

def initialize_variable(sz, val,device):
   d1 = np.empty(sz)
   d1[:] = val
   return torch.from_numpy(d1).to(device)

def render_all(mesh, opt):
    measurement_num = opt.lighting.shape[0]
    transient = np.empty((measurement_num, opt.max_distance_bin))
    for i in range(measurement_num):
        transient[i] = render(mesh, opt.lighting[i,:], opt.sensor[i,:], opt.lighting_normal[i,:], opt.sensor_normal[i,:], opt) 
    return transient

def render(mesh, lighting, sensor, lighting_normal, sensor_normal, opt):
    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num, 1)))
    
    phi = 2*math.pi*np.random.rand(opt.sample_num)
    theta = np.arccos(np.random.rand(opt.sample_num))
    R = np.zeros([3,3])
    rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)

    direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
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
    fn = np.array(mesh.fn)
    normalMap[idx,:] = fn[np.hstack(fid[idx]),:]

    v11[idx] = lighting[0] - intersection_p[idx,0]
    d1[idx] = np.abs(np.divide(v11[idx], direction[idx,0]))

    v2[idx,:] = sensor - intersection_p[idx,:]
    d2[idx] = np.sqrt(np.sum(v2[idx,:]**2, axis = 1)) 
    v2[idx,:] = element_divide2_np(v2[idx,:], d2[idx])

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

    inds = list(compress(range(opt.sample_num), distance_bin<opt.max_distance_bin))
    intensity[inds] = np.divide(cos_theta2[inds], d2[inds]**2)

    u = np.unique(distance_bin[inds])

    for x in u:
        angular_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num
    return angular_transient

def render_transient(mesh, lighting, sensor, lighting_normal, sensor_normal, opt, angular_transient):
    source = igl.eigen.MatrixXd(np.tile(lighting, (opt.sample_num, 1)))
    
    phi = 2*math.pi*np.random.rand(opt.sample_num)
    theta = np.arccos(np.random.rand(opt.sample_num))
    R = np.zeros([3,3])
    rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)

    direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
    n = igl.eigen.MatrixXd(direction)

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
    fn = np.array(mesh.fn)
    normalMap[idx,:] = fn[np.hstack(fid[idx]),:]

    v11[idx] = lighting[0] - intersection_p[idx,0]
    d1[idx] = np.abs(np.divide(v11[idx], direction[idx,0]))

    v2[idx,:] = sensor - intersection_p[idx,:]
    d2[idx] = np.sqrt(np.sum(v2[idx,:]**2, axis = 1)) 
    v2[idx,:] = element_divide2_np(v2[idx,:], d2[idx])

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

    inds = list(compress(range(opt.sample_num), distance_bin<opt.max_distance_bin))
    intensity[inds] = np.divide(cos_theta2[inds], d2[inds]**2)

    u = np.unique(distance_bin[inds])

    for x in u:
        angular_transient[x.astype(int)] += sum(intensity[distance_bin == x])

    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num

def render_differentiable_transient(mesh, direction, triangleIndexMap, lighting, sensor, lighting_normal, sensor_normal, opt, device, angular_transient):

    triangleIndexMap = torch.from_numpy(triangleIndexMap).long().to(device)
    direction = torch.from_numpy(direction).to(device)
    lighting = torch.from_numpy(lighting).to(device)
    sensor = torch.from_numpy(sensor).to(device)

    d1 = initialize_variable(opt.sample_num, -1, device)
    d2 = initialize_variable(opt.sample_num, np.nan, device)
    intensity = initialize_variable(opt.sample_num, np.nan, device)
    uMap = initialize_variable(opt.sample_num, np.nan, device)
    vMap = initialize_variable(opt.sample_num, np.nan, device)
    cos_theta2 = initialize_variable(opt.sample_num, 0, device)
    intersection_p = initialize_variable((opt.sample_num,3), np.nan, device)
    normalMap = initialize_variable((opt.sample_num,3), np.nan, device)
    v2 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e1 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e2 = initialize_variable((opt.sample_num,3), np.nan, device)
    fn = initialize_variable((opt.sample_num,3), np.nan, device)
    fn_len = initialize_variable(opt.sample_num, np.nan, device)

    distance_bin = torch.LongTensor(opt.sample_num).fill_(opt.max_distance_bin+1).to(device)

    inds = torch.squeeze((triangleIndexMap > -1).data.nonzero())
    for i in inds:
        uMap[i], vMap[i], d1[i] = intersect_ray_mesh_one_direction(mesh, torch.squeeze(direction[i,:]), torch.squeeze(mesh.f[triangleIndexMap[i],:]), lighting)

    inds = torch.squeeze((d1 > 0).data.nonzero())

    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)
    
    data = element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:]) + element_multiply2(uMap[inds], mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:]) + element_multiply2(vMap[inds], mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:])
    intersection_p.index_copy_(0, inds, data)


    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = torch.sqrt(torch.sum(v2[inds,:]**2, 1))
    v2[inds,:] = element_divide2(v2[inds,:], d2[inds])

    tmp_e1[inds,:] = mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 
    tmp_e2[inds,:] = mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 

    fn[inds,:] = torch.cross(tmp_e1[inds,:], tmp_e2[inds,:], 1) 
    fn_len[inds] = torch.sqrt(torch.sum(fn[inds,:]**2,1)) 
    normalMap[inds,:] = element_divide2(fn[inds,:], fn_len[inds])
    cos_theta2[inds] = torch.sum(torch.mul(normalMap[inds,:], v2[inds,:]),1)
    index = torch.squeeze((cos_theta2<0).data.nonzero())
    if index.dim() != 0:
        if len(index) != 0:
            cos_theta2 = cos_theta2.index_fill(0, index, 0)

    distance_bin[inds] = torch.ceil((d1[inds]+d2[inds])/opt.distance_resolution).long() -1
    inds = torch.squeeze((distance_bin<opt.max_distance_bin).data.nonzero())
    if inds.dim() != 0:
        if len(inds) != 0:
          val = torch.div(cos_theta2.index_select(0,inds), d2.index_select(0,inds)**2)
          intensity.index_copy_(0,inds,val)
          angular_transient.index_add_(0, distance_bin[inds],  intensity[inds])
    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num
    return angular_transient

def render_differentiable(mesh, direction, triangleIndexMap, lighting, sensor, lighting_normal, sensor_normal, opt, device):
    angular_transient = torch.DoubleTensor(opt.max_distance_bin).fill_(0).to(device)

    triangleIndexMap = torch.from_numpy(triangleIndexMap).long().to(device)
    direction = torch.from_numpy(direction).to(device)
    lighting = torch.from_numpy(lighting).to(device)
    sensor = torch.from_numpy(sensor).to(device)

    d1 = initialize_variable(opt.sample_num, -1, device)
    d2 = initialize_variable(opt.sample_num, np.nan, device)
    intensity = initialize_variable(opt.sample_num, np.nan, device)
    uMap = initialize_variable(opt.sample_num, np.nan, device)
    vMap = initialize_variable(opt.sample_num, np.nan, device)
    cos_theta2 = initialize_variable(opt.sample_num, 0, device)
    intersection_p = initialize_variable((opt.sample_num,3), np.nan, device)
    normalMap = initialize_variable((opt.sample_num,3), np.nan, device)
    v2 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e1 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e2 = initialize_variable((opt.sample_num,3), np.nan, device)
    fn = initialize_variable((opt.sample_num,3), np.nan, device)
    fn_len = initialize_variable(opt.sample_num, np.nan, device)

    distance_bin = torch.LongTensor(opt.sample_num).fill_(opt.max_distance_bin+1).to(device)

    inds = torch.squeeze((triangleIndexMap > -1).data.nonzero())

    for i in inds:
        uMap[i], vMap[i], d1[i] = intersect_ray_mesh_one_direction(mesh, torch.squeeze(direction[i,:]), torch.squeeze(mesh.f[triangleIndexMap[i],:]), lighting)

    inds = torch.squeeze((d1 > 0).data.nonzero())

    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)
    
    data = element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:]) + element_multiply2(uMap[inds], mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:]) + element_multiply2(vMap[inds], mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:])
    intersection_p.index_copy_(0, inds, data)


    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = torch.sqrt(torch.sum(v2[inds,:]**2, 1))
    v2[inds,:] = element_divide2(v2[inds,:], d2[inds])

    tmp_e1[inds,:] = mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 
    tmp_e2[inds,:] = mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 

    fn[inds,:] = torch.cross(tmp_e1[inds,:], tmp_e2[inds,:], 1) 
    fn_len[inds] = torch.sqrt(torch.sum(fn[inds,:]**2,1)) 
    normalMap[inds,:] = element_divide2(fn[inds,:], fn_len[inds])
    cos_theta2[inds] = torch.sum(torch.mul(normalMap[inds,:], v2[inds,:]),1)
    index = torch.squeeze((cos_theta2<0).data.nonzero())
    if index.dim() != 0:
        if len(index) != 0:
            cos_theta2 = cos_theta2.index_fill(0, index, 0)

    distance_bin[inds] = torch.ceil((d1[inds]+d2[inds])/opt.distance_resolution).long() -1
    inds = torch.squeeze((distance_bin<opt.max_distance_bin).data.nonzero())
    if inds.dim() != 0:
        if len(inds) != 0:
          val = torch.div(cos_theta2.index_select(0,inds), d2.index_select(0,inds)**2)
          intensity.index_copy_(0,inds,val)
          angular_transient.index_add_(0, distance_bin[inds],  intensity[inds])
    angular_transient *= 2*math.pi
    angular_transient /= opt.sample_num
    #smooth = torch.reshape(torch.from_numpy(np.array([0.2, 0.6, 0.2])), (1,1,3))
    #angular_transient = torch.reshape(angular_transient,(1,1,opt.max_distance_bin))
    #angular_transient = torch.nn.functional.conv1d(angular_transient, smooth, None, 1, 1)
    #angular_transeint = torch.reshape(angular_transient, opt.max_distance_bin)
    return angular_transient

def find_grad(index, measurement, mesh, mesh_optimization, opt, render_opt, device):
	transient = render(mesh, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], render_opt)
	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal[index,:])
	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
	triangleIndexMap = find_triangle(mesh, direction, opt.lighting[index,:], opt.sensor[index,:])

	transient_differentiable = render_differentiable(mesh_optimization, direction, triangleIndexMap, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt, device)

	weight = -2* torch.from_numpy(measurement[index,:] - transient).to(device)
	transient_differentiable.backward(weight)

def render_differentiable_rand_direction(mesh, mesh_optimization, lighting, sensor, lighting_normal, sensor_normal, opt, device, transient_differentiable):
	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), lighting_normal)
	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
	triangleIndexMap = find_triangle(mesh, direction, lighting, sensor)

	render_differentiable_transient(mesh_optimization, direction, triangleIndexMap, lighting, sensor, lighting_normal, sensor_normal, opt, device, transient_differentiable)

def loss_func_parallel2(index, measurement, mesh, mesh_optimization, opt, render_opt, device):
	thread_collection = []
	
	transient = np.zeros(render_opt.max_distance_bin)
	t = threading.Thread(target=render_transient, args=[mesh, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], render_opt, transient])
	t.start()
	thread_collection.append(t)


	transient_differentiable = torch.DoubleTensor(opt.max_distance_bin).fill_(0).to(device)
	t = threading.Thread(target=render_differentiable_rand_direction, args=[mesh, mesh_optimization, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt, device, transient_differentiable])
	t.start()
	thread_collection.append(t)
	

	for t in thread_collection:
	  t.join()

	#weight = -2*torch.mul(torch.from_numpy(measurement[index,:] - transient), torch.from_numpy(measurement[index,:]+0.002)).to(device)
	weight = -2*torch.from_numpy(measurement[index,:] - transient).to(device)
	first_photon = measurement[index,:].nonzero()[0][0]
	weight[first_photon:] *= -1
	return torch.mul(weight, transient_differentiable).sum()

def render_intensity_differentiable(mesh, direction, triangleIndexMap, difference, lighting, sensor, lighting_normal, sensor_normal, opt, device):

    triangleIndexMap = torch.from_numpy(triangleIndexMap).long().to(device)
    direction = torch.from_numpy(direction).to(device)
    lighting = torch.from_numpy(lighting).to(device)
    sensor = torch.from_numpy(sensor).to(device)

    d1 = initialize_variable(opt.sample_num, -1, device)
    d2 = initialize_variable(opt.sample_num, np.nan, device)
    intensity = initialize_variable(opt.sample_num, np.nan, device)
    uMap = initialize_variable(opt.sample_num, np.nan, device)
    vMap = initialize_variable(opt.sample_num, np.nan, device)
    cos_theta2 = initialize_variable(opt.sample_num, 0, device)
    intersection_p = initialize_variable((opt.sample_num,3), np.nan, device)
    normalMap = initialize_variable((opt.sample_num,3), np.nan, device)
    v2 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e1 = initialize_variable((opt.sample_num,3), np.nan, device)
    tmp_e2 = initialize_variable((opt.sample_num,3), np.nan, device)
    fn = initialize_variable((opt.sample_num,3), np.nan, device)
    fn_len = initialize_variable(opt.sample_num, np.nan, device)
    w = initialize_variable(opt.sample_num, np.nan, device)

    distance_bin = torch.LongTensor(opt.sample_num).fill_(opt.max_distance_bin+1).to(device)

    inds = torch.squeeze((triangleIndexMap > -1).data.nonzero())

    for i in inds:
        uMap[i], vMap[i], d1[i] = intersect_ray_mesh_one_direction(mesh, torch.squeeze(direction[i,:]), torch.squeeze(mesh.f[triangleIndexMap[i],:]), lighting)

    inds = torch.squeeze((d1 > 0).data.nonzero())

    triangleIndexMap_input = torch.index_select(triangleIndexMap, 0, inds)
    
    data = element_multiply2(1-uMap[inds] - vMap[inds], mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:]) + element_multiply2(uMap[inds], mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:]) + element_multiply2(vMap[inds], mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:])
    intersection_p.index_copy_(0, inds, data)


    v2[inds,:] = sensor - intersection_p[inds,:]
    d2[inds] = torch.sqrt(torch.sum(v2[inds,:]**2, 1))
    v2[inds,:] = element_divide2(v2[inds,:], d2[inds])

    tmp_e1[inds,:] = mesh.v[torch.index_select(mesh.f[:,1],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 
    tmp_e2[inds,:] = mesh.v[torch.index_select(mesh.f[:,2],0,triangleIndexMap_input),:] - mesh.v[torch.index_select(mesh.f[:,0],0,triangleIndexMap_input),:] 

    fn[inds,:] = torch.cross(tmp_e1[inds,:], tmp_e2[inds,:], 1) 
    fn_len[inds] = torch.sqrt(torch.sum(fn[inds,:]**2,1)) 
    normalMap[inds,:] = element_divide2(fn[inds,:], fn_len[inds])
    cos_theta2[inds] = torch.sum(torch.mul(normalMap[inds,:], v2[inds,:]),1)
    index = torch.squeeze((cos_theta2<0).data.nonzero())
    if index.dim() != 0:
        if len(index) != 0:
            cos_theta2 = cos_theta2.index_fill(0, index, 0)

    distance_bin[inds] = torch.ceil((d1[inds]+d2[inds])/opt.distance_resolution).long() -1
    inds = torch.squeeze((distance_bin<opt.max_distance_bin).data.nonzero())
    if inds.dim() != 0:
        if len(inds) != 0:
          w[inds] = difference.index_select(0,distance_bin[inds])
          val1 = torch.div(cos_theta2.index_select(0,inds), d2.index_select(0,inds)**2)
          val2 = torch.mul(w.index_select(0,inds), val1)
          #val2 = torch.mul(difference.index_select(0,inds), val1)
          intensity.index_copy_(0,inds,val2)
    return torch.sum(intensity)*2*math.pi/opt.sample_num

def filt(center, sigma, length, device):
    gauss_filt = torch.ones(length).double().to(device)
    #gauss_filt = np.zeros(length)
    #gauss_filt[center-sigma:center+sigma] = 1
    #gauss_filt = torch.from_numpy(gauss_filt).double().to(device)
    #gauss_filt = torch.from_numpy(np.arange(length))
    #gauss_filt = np.exp(-(gauss_filt-center)**2/2/(sigma**2))/ math.sqrt(2*math.pi)/sigma
    return gauss_filt

def loss_func_parallel(index, measurement, mesh, mesh_optimization, opt, render_opt, device):
	thread_collection = []
	
	transient = np.zeros(render_opt.max_distance_bin)
	t = threading.Thread(target=render_transient, args=[mesh, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], render_opt, transient])
	t.start()
	thread_collection.append(t)


	transient_differentiable = torch.DoubleTensor(opt.max_distance_bin).fill_(0).to(device)
	t = threading.Thread(target=render_differentiable_rand_direction, args=[mesh, mesh_optimization, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt, device, transient_differentiable])
	t.start()
	thread_collection.append(t)
	

	for t in thread_collection:
	  t.join()

	#weight = -2*torch.mul(torch.from_numpy(measurement[index,:] - transient), torch.from_numpy(measurement[index,:]+0.002)).to(device)
	weight = -2*torch.from_numpy(measurement[index,:] - transient).to(device)
	return torch.mul(weight, transient_differentiable).sum()

def loss_func(index, measurement, mesh, mesh_optimization, opt, render_opt, device):
	transient = render(mesh, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], render_opt)
	phi = 2*math.pi*np.random.rand(opt.sample_num)
	theta = np.arccos(np.random.rand(opt.sample_num))
	R = np.zeros([3,3])
	rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal[index,:])
	direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
	triangleIndexMap = find_triangle(mesh, direction, opt.lighting[index,:], opt.sensor[index,:])


	transient_differentiable = render_differentiable(mesh_optimization, direction, triangleIndexMap, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt, device)

	weight = -2* torch.from_numpy(measurement[index,:] - transient).to(device)
	return torch.mul(weight, transient_differentiable).sum()


def loss_fun_weighted_time(index, measurement, mesh, mesh_optimization, opt, render_opt, device):
    transient = render(mesh, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], render_opt)
    phi = 2*math.pi*np.random.rand(opt.sample_num)
    theta = np.arccos(np.random.rand(opt.sample_num))
    R = np.zeros([3,3])
    rotation_matrix.R_2vect(R, np.array([0,0,1]), opt.lighting_normal[index,:])
    direction = np.dot(np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta))).T, R.T)
    #triangleIndexMap, distance_bin = find_triangle(mesh, direction, opt.lighting[index,:], opt.sensor[index,:], opt)
    triangleIndexMap = find_triangle(mesh, direction, opt.lighting[index,:], opt.sensor[index,:], opt)
    
    w = torch.ones((1,1,2*opt.w_width+1)).to(device).double()
    difference = torch.reshape(-2 * torch.from_numpy(measurement[index,:] - transient).to(device), (1,1,opt.max_distance_bin))
    difference = torch.nn.functional.conv1d(difference, w, None, 1, opt.w_width)
    difference = torch.nn.functional.conv1d(difference, w, None, 1, opt.w_width)
    weight = torch.squeeze(difference)
    #difference = torch.squeeze(difference)
    
    #weight = difference.index_select(0,torch.from_numpy(distance_bin).long())

    return render_intensity_differentiable(mesh_optimization, direction, triangleIndexMap, weight, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt, device)

        
def evaluate_L2(gt_transient, mesh, render_opt):
  transient = render_all(mesh, render_opt)
  return np.linalg.norm(transient - gt_transient), transient 

def evaluate_vertex_L2(gt_mesh, mesh):
  return np.linalg.norm( np.array(gt_mesh.v - mesh.v)) 
