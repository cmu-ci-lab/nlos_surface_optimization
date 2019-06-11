import numpy as np
import math
import time
from itertools import compress
from multiprocessing import Pool

from functools import partial

from scipy.spatial import Delaunay
import pyigl as igl

def space_carving_initialization(mesh, space_carving_mesh, opt):
	[x, y] = np.meshgrid(np.linspace(-2.2, 1.8, 7), np.linspace(-2.2, 1.8, 7))
	x = np.concatenate(x)
	y = np.concatenate(y)
	v = np.vstack((x,y,np.ones_like(x)*opt.max_distance_bin*opt.distance_resolution/2)).T
	tri = Delaunay(v[:,0:2])

	mesh.f = igl.eigen.MatrixXi(tri.simplices[:,[0,2,1]])
	mesh.v = igl.eigen.MatrixXd(v)
		    
	direction = igl.eigen.MatrixXd(np.tile(np.array([0,0,1.0]), (mesh.v.rows(), 1)))
	barycoord = igl.eigen.MatrixXd()
	barycoord = igl.embree.line_mesh_intersection(mesh.v, direction, space_carving_mesh.v, space_carving_mesh.f)
	point = igl.barycentric_to_global(space_carving_mesh.v, space_carving_mesh.f, barycoord)

	fid = np.array(barycoord.col(0))
	for x in list(compress(range(mesh.v.rows()), fid != -1)):
	  mesh.v.setRow(x, point.row(x))
	  #mesh.v.setRow(x, igl.eigen.MatrixXd(np.array(point.row(x))-np.array([0,0,np.random.normal(0,0.1)])))


def space_carving_projection(mesh_optimization, space_carving_mesh):
    v = igl.eigen.MatrixXd(mesh_optimization.v.data.numpy())
    direction = igl.eigen.MatrixXd.Zero(v.rows(),3)
    direction.setCol(2, igl.eigen.MatrixXd.Ones(v.rows(),1))

    barycoord = igl.eigen.MatrixXd()
    barycoord = igl.embree.line_mesh_intersection(v, direction, space_carving_mesh.v, space_carving_mesh.f)
    point = igl.barycentric_to_global(space_carving_mesh.v, space_carving_mesh.f, barycoord)   
    z = np.array(point.col(2)) 
    z_original = np.array(v.col(2))
    for x in list(compress(range(v.rows()), z_original < z)):
        v.setRow(x, point.row(x))
    return v

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()
  doublearea = igl.eigen.MatrixXd()

def grad_fun(index, gt_transient, transient, v, f, opt):
  mesh = MESH()
  mesh.v = igl.eigen.MatrixXd(v)
  mesh.f = igl.eigen.MatrixXi(f)
  igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
  igl.doublearea(mesh.v, mesh.f, mesh.doublearea)
  gradient = grad_collocate(index, gt_transient, transient, mesh, opt)
  return gradient
  
def grad_parallel(gt_transient, transient, mesh, opt):
  v = np.array(mesh.v)
  f = np.array(mesh.f)
  myfunc = partial(grad_fun, gt_transient=gt_transient, transient=transient, v=v, f=f, opt=opt)

  index = np.arange(gt_transient.shape[0])    
  with Pool(processes=opt.thread_num) as p:  
    result = p.map(myfunc, index)
    grad = sum(result)
  return grad



def evaluate_L2(gt_transient, mesh, render_opt):
  transient = render_all(mesh, render_opt)
  return np.linalg.norm(transient - gt_transient)**2, transient 

def evaluate_L2_collocate(gt_transient, mesh, render_opt):
  transient = render_all_collocate(mesh, render_opt)
  return np.linalg.norm(transient - gt_transient)**2, transient 

def evaluate_smooth_L2_collocate(gt_transient, mesh, render_opt, smooth_opt):
  transient = render_all_collocate(mesh, render_opt)
  #barycoord = random_barycoord(mesh, render_opt.sample_num)
  #tic = time.time() 
  #new_transient = new_render_all_collocate(mesh, render_opt, barycoord)
  #print('new transient %f'%(time.time() - tic))
  
  #tic = time.time()
  #measurement_num = render_opt.lighting.shape[0]
  #transient = np.empty((measurement_num, render_opt.max_distance_bin))
  #for i in range(measurement_num):
  #    transient[i] = mesh_sampling_collocate(mesh, barycoord, render_opt.lighting[i,:], render_opt.lighting_normal[i,:], render_opt)
  #print('transient %f'%(time.time() -tic))
  #print('diff %f'% np.linalg.norm(new_transient-transient))  

  

  s = np.array(mesh.v.col(2))
  s = np.reshape(s, smooth_opt.v_shape)
  Dx_s = s[:,1:] - s[:,0:-1]
  Dy_s = s[1:,:] - s[0:-1,:]

 
  w = np.ones(2*render_opt.w_width+1)/(2*render_opt.w_width+1)
  difference = transient - gt_transient
  for i in range(difference.shape[0]):
    difference[i,:] = np.convolve(difference[i,:], w, 'same')
  
 
  L1 = np.linalg.norm(difference)**2 
  L2 = (np.linalg.norm(Dx_s)**2 + np.linalg.norm(Dy_s)**2)*smooth_opt.weight/2
  return L1+L2, transient, L1

def new_render_all_collocate(mesh,opt,barycoord):
    #barycoord = random_barycoord(mesh, opt.sample_num)
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


def render_all_fun(i, v, f, opt):
  mesh = MESH()
  mesh.v = igl.eigen.MatrixXd(v)
  mesh.f = igl.eigen.MatrixXi(f)
  igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
  igl.doublearea(mesh.v, mesh.f, mesh.doublearea)
  transient = mesh_sampling_collocate(mesh, opt.lighting[i,:], opt.lighting_normal[i,:], opt)
  return transient

def render_all_collocate(mesh, opt):
  v = np.array(mesh.v)
  f = np.array(mesh.f)
  myfunc = partial(render_all_fun, v=v, f=f, opt=opt)

  index = np.arange(opt.lighting.shape[0])    
  with Pool(processes=opt.thread_num) as p:  
    result = p.map(myfunc, index)
    transient = np.asarray(result)
  return transient


    #measurement_num = opt.lighting.shape[0]
    #transient = np.empty((measurement_num, opt.max_distance_bin))
    #for i in range(measurement_num):
    #    transient[i] = mesh_sampling_collocate(mesh, opt.lighting[i,:], opt.lighting_normal[i,:], opt)
    #    transient[i] = amortized_mesh_sampling_collocate(mesh, opt.lighting[i,:], opt.lighting_normal[i,:], opt)
    #return transient

def render_all(mesh, opt):
    measurement_num = opt.lighting.shape[0]
    transient = np.empty((measurement_num, opt.max_distance_bin))
    for i in range(measurement_num):
        barycoord = random_barycoord(mesh, opt.sample_num)
        transient[i] = mesh_sampling(mesh, barycoord, opt.lighting[i,:], opt.sensor[i,:], opt.lighting_normal[i,:], opt.sensor_normal[i,:], opt)
    return transient

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

def smooth_grad(mesh, smooth_opt):
    s = np.array(mesh.v.col(2))
    s = np.reshape(s, smooth_opt.v_shape)
    Dx_s = s[:,1:] - s[:,0:-1]
    x = np.hstack((-Dx_s, np.zeros((smooth_opt.v_shape[0],1))))
    x[:,1:] += Dx_s

    Dy_s = s[1:,:] - s[0:-1,:]
    y = np.vstack((-Dy_s,np.zeros((1,smooth_opt.v_shape[1]))))
    y[1:,:] += Dy_s

    grad_z = (x + y).flatten()

    grad = np.zeros((mesh.v.rows(),3))
    grad[:,2] += grad_z*smooth_opt.weight

    return grad


def smooth_grad_collocate(index, measurement, mesh, opt, render_opt, smooth_opt):
    
    s = np.array(mesh.v.col(2))
    s = np.reshape(s, smooth_opt.v_shape)
    Dx_s = s[:,1:] - s[:,0:-1]
    x = np.hstack((-Dx_s, np.zeros((smooth_opt.v_shape[0],1))))
    x[:,1:] += Dx_s

    Dy_s = s[1:,:] - s[0:-1,:]
    y = np.vstack((-Dy_s,np.zeros((1,smooth_opt.v_shape[1]))))
    y[1:,:] += Dy_s

    grad_z = (x + y).flatten()

    grad = grad_collocate(index, measurement, mesh, opt, render_opt)
    grad[:,2] += grad_z*smooth_opt.weight

    return grad



def grad_collocate(index, measurement, transient, mesh, opt):
    transient = transient[index,:]
    
    barycoord = random_barycoord(mesh, opt.sample_num)

    mesh_grad = new_mesh_grad_sampling_collocate(mesh, barycoord, opt.lighting[index,:], opt.lighting_normal[index,:], opt)
    #old_mesh_grad = new_mesh_grad_sampling_collocate(mesh, barycoord, opt.lighting[index,:], opt.lighting_normal[index,:], opt)
    #print('diff %f'%np.linalg.norm(mesh_grad-old_mesh_grad))
    w = np.ones(2*opt.w_width+1)/(2*opt.w_width+1)
    difference = -2 * (measurement[index,:] - transient)
    difference = np.convolve(difference, w, 'same')
    difference = np.convolve(difference, w, 'same')

    mesh_grad = np.multiply(mesh_grad, np.vstack(difference))
    mesh_grad = np.reshape(np.sum(mesh_grad,0), (mesh.v.rows(),3))
    mesh_grad[:,0:2] = 0
    return mesh_grad 

def grad(index, measurement, mesh, opt, render_opt):
    barycoord = random_barycoord(mesh, render_opt.sample_num)
    transient = mesh_sampling(mesh, barycoord, render_opt.lighting[index,:], render_opt.sensor[index,:], render_opt.lighting_normal[index,:], render_opt.sensor_normal[index,:], render_opt)
    
    barycoord = random_barycoord(mesh, opt.sample_num)
    mesh_grad = mesh_grad_sampling(mesh, barycoord, opt.lighting[index,:], opt.sensor[index,:], opt.lighting_normal[index,:], opt.sensor_normal[index,:], opt)

    w = np.ones(2*opt.w_width+1)
    difference = -2 * (measurement[index,:] - transient)
    difference = np.convolve(difference, w, 'same')
    difference = np.convolve(difference, w, 'same')

    mesh_grad = np.multiply(mesh_grad, np.vstack(difference))
    mesh_grad = np.reshape(np.sum(mesh_grad,0), (mesh.v.rows(),3))
    mesh_grad[:,0:2] = 0
    return mesh_grad 

def new_mesh_grad_sampling_collocate(mesh, barycoord, lighting, lighting_normal, opt):
    fn = np.array(mesh.fn)
    face_area = np.array(mesh.doublearea)/2
    grad = np.zeros((opt.max_distance_bin, 3*mesh.v.rows()))

    sample_num = opt.sample_num
    fid = np.array(barycoord.col(0))
    u = np.array(barycoord.col(1))
    v = np.array(barycoord.col(2))
    phi = 1-u-v
            
    sample_point = igl.barycentric_to_global(mesh.v, mesh.f, barycoord)
    source = igl.eigen.MatrixXd(np.tile(lighting, (sample_num,1)))
    v1 = source-sample_point
    d1 = np.array(v1.rowwiseNorm())
    v1 = v1.rowwiseNormalized()
    sample_point = np.array(sample_point)

    barycoord1 = igl.eigen.MatrixXd()
    barycoord1 = igl.embree.line_mesh_intersection(source, -v1, mesh.v, mesh.f)

    v1 = np.array(v1)
    fid1 = np.array(barycoord1.col(0))

    visible_idx = np.hstack((fid1 == fid))
    f_idx = np.hstack(fid[visible_idx]).astype(int)
    normalMap = np.empty((sample_num, 3))
    normalMap[visible_idx,:] = fn[f_idx,:]
    cos_theta1 = np.zeros(sample_num)
    cos_theta1[visible_idx] = np.einsum('ij,ij->i', normalMap[visible_idx,:], np.array(v1)[visible_idx,:])
    cos_theta1[cos_theta1<0] = 0 

    d1_new = np.squeeze(d1[visible_idx])
    distance_bin = np.ones(sample_num) * (opt.max_distance_bin + 1)
    distance_bin[visible_idx] = np.ceil((d1_new *2)/opt.distance_resolution) -1
    visible_idx = (distance_bin < opt.max_distance_bin)
    fid[distance_bin >= opt.max_distance_bin] = -1    
 
    intensity = np.empty(sample_num)
    d1_new = np.squeeze(d1[visible_idx])
    intensity[visible_idx] = np.divide(cos_theta1[visible_idx]**2, (d1_new**4))

    face_num = mesh.f.rows()
    for vertex in range(mesh.v.rows()):

        related_face = list(compress(range(face_num), np.sum(np.array(mesh.f) == vertex, 1) > 0 ))

        for f in related_face:
            face = np.array(mesh.f.row(f))[0]
            vertex_idx = list(compress(range(3), face == vertex))
            vertex_idx = vertex_idx[0]

            ind = np.squeeze(fid == f) 
            sample_num = np.sum(ind)
            if sample_num == 0:
               continue

            u_new = u[ind,:]
            v_new = v[ind,:]
            phi_new = phi[ind]

            sample_point_new = sample_point[ind,:]
            d1_new = d1[ind]
            v1_new = v1[ind,:]           
 
            normalMap_new = fn[f,:]
           
            cos_theta1_new = cos_theta1[ind] 
            distance_bin_new = distance_bin[ind]            
            intensity_new = intensity[ind]            

            t1_tmp1 = 6*v1_new*np.vstack(cos_theta1_new**2) - 2*normalMap_new*np.vstack(cos_theta1_new)
            t1 = t1_tmp1/np.vstack(d1_new**5)
            
            t2 = normalMap_new*np.vstack(intensity_new) 

            gn_tmp1 = 2*v1_new*np.vstack(cos_theta1_new)
            gn_tmp = gn_tmp1/np.vstack(d1_new**4)
            
            cos_tmp = np.dot(gn_tmp, normalMap_new)
            gn = gn_tmp - normalMap_new*np.vstack(cos_tmp)
            
            t2 = (t2 + gn)/(2*face_area[f])
            if vertex_idx == 0:
                e = np.array(mesh.v.row(face[2]) - mesh.v.row(face[1]))
                g = t1*phi_new + np.cross(t2, e)
            elif vertex_idx == 1:
                e = np.array(mesh.v.row(face[0]) - mesh.v.row(face[2]))
                g = t1*u_new + np.cross(t2, e)
            else: 
                e = np.array(mesh.v.row(face[1]) - mesh.v.row(face[0]))
                g = t1*v_new + np.cross(t2, e)

            unique_bin = np.unique(distance_bin_new)
            
            for x in unique_bin:
                grad[x.astype(int),3*vertex:3*(vertex+1)] += np.sum(g[distance_bin_new == x,:],0)*face_area[f]

    grad /= opt.sample_num
    return grad

def mesh_grad_sampling_collocate(mesh, barycoord, lighting, lighting_normal, opt):
    fn = np.array(mesh.fn)
    face_area = np.array(mesh.doublearea)/2
    grad = np.zeros((opt.max_distance_bin, 3*mesh.v.rows()))

    for vertex in range(mesh.v.rows()):

        related_face = list(compress(range(mesh.f.rows()), np.sum(np.array(mesh.f) == vertex, 1) > 0 ))

        for f in related_face:
            face = np.array(mesh.f.row(f))[0]
            vertex_idx = list(compress(range(3), face == vertex))
            vertex_idx = vertex_idx[0]
            ind = np.squeeze(np.array(barycoord.col(0)) == f) 
            
            sample_num = np.sum(ind)
            if sample_num == 0:
               continue

            u = np.array(barycoord.col(1))[ind,:]
            v = np.array(barycoord.col(2))[ind,:]
            phi = 1-u-v

            
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
            if len(idx) == 0:
                continue

            normalMap = fn[f,:]
            u_new = u[idx]
            v_new = v[idx]
            phi_new = phi[idx]
            
            v1_new = np.array(v1)[idx,:]
            cos_theta1 = np.dot(v1_new, normalMap)
            cos_theta1[cos_theta1<0] = 0
            d1_new = np.squeeze(d1[idx])
            distance_bin = np.ceil((d1_new *2)/opt.distance_resolution) -1
            intensity = np.divide(cos_theta1**2, (d1_new**4))
            if isinstance(distance_bin, np.float64):
                if distance_bin >= opt.max_distance_bin:
                   continue
                else:
                   intensity = np.array(intensity)
                   d1_new = np.array([d1_new])
                   cos_theta1 = np.array(cos_theta1)
                   phi_new = np.array(phi_new)
                   u_new = np.array(u_new)
                   v_new = np.array(v_new)
                   distance_bin = np.array([distance_bin])
                   idx = [0]
            else:
                idx = list(compress(range(len(distance_bin)), distance_bin < opt.max_distance_bin))
                if len(idx) == 0:
                   continue
            t1_tmp1 = 6*v1_new[idx,:]*np.vstack(cos_theta1[idx]**2) - 2*normalMap*np.vstack(cos_theta1[idx])
            t1 = t1_tmp1/np.vstack(d1_new[idx]**5)

            
            t2 = normalMap*np.vstack(intensity[idx]) 

            gn_tmp1 = 2*v1_new[idx,:]*np.vstack(cos_theta1[idx])
            gn_tmp = gn_tmp1/np.vstack(d1_new[idx]**4)
            
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

def mesh_grad_sampling(mesh, barycoord, lighting, sensor, lighting_normal, sensor_normal, opt):
    fn = np.array(mesh.fn)
    face_area = np.array(mesh.doublearea)/2
    grad = np.zeros((opt.max_distance_bin, 3*mesh.v.rows()))

    for vertex in range(mesh.v.rows()):

        related_face = list(compress(range(mesh.f.rows()), np.sum(np.array(mesh.f) == vertex, 1) > 0 ))

        for f in related_face:
            face = np.array(mesh.f.row(f))[0]
            vertex_idx = list(compress(range(3), face == vertex))
            vertex_idx = vertex_idx[0]
            ind = np.squeeze(np.array(barycoord.col(0)) == f) 
            
            sample_num = np.sum(ind)
            if sample_num == 0:
               continue

            u = np.array(barycoord.col(1))[ind,:]
            v = np.array(barycoord.col(2))[ind,:]
            phi = 1-u-v

            
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
            if len(idx) == 0:
                continue

            source = igl.eigen.MatrixXd(np.tile(sensor, (len(idx),1)))

            sample_point_subset = igl.eigen.MatrixXd(np.array(sample_point)[idx,:])
            v2 = source - sample_point_subset
            d2 = np.array(v2.rowwiseNorm())
            v2 = v2.rowwiseNormalized()

            barycoord2 = igl.eigen.MatrixXd()
            barycoord2 = igl.embree.line_mesh_intersection(source, -v2, mesh.v, mesh.f)
            fid2 = np.array(barycoord2.col(0))
            idx1 = list(compress(idx, fid2 == f))
            if len(idx1) == 0:
                continue

            idx2 = list(compress(range(len(fid2)), fid2 == f))
            
            normalMap = fn[f,:]
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
            intensity = np.divide(cos_theta1*cos_theta2, (d1_new**2)*(d2_new**2))
            if isinstance(distance_bin, np.float64):
                if distance_bin >= opt.max_distance_bin:
                   continue
                else:
                   intensity = np.array(intensity)
                   d1_new = np.array([d1_new])
                   d2_new = np.array([d2_new])
                   cos_theta1 = np.array(cos_theta1)
                   cos_theta2 = np.array(cos_theta2)
                   phi_new = np.array(phi_new)
                   u_new = np.array(u_new)
                   v_new = np.array(v_new)
                   distance_bin = np.array([distance_bin])
                   idx = [0]
            else:
                idx = list(compress(range(len(distance_bin)), distance_bin < opt.max_distance_bin))
                if len(idx) == 0:
                   continue
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

def amortized_mesh_sampling_collocate(mesh, lighting, lighting_normal, opt):
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
