import numpy as np
import scipy.io
import sys, os
#sys.path.insert(0,'/Users/chiayint/Documents/research/libigl/python/')
sys.path.insert(0,'/home/chiayint/research/libigl/python/')
import math
import time
import argparse
from itertools import compress

import rotation_matrix
import rendering
import torch
from torch import optim

import pyigl as igl

class OPT:
  def __init__(self, sample_num=2500):
    self.sample_num = sample_num
  max_distance_bin = 1146
  distance_resolution = 5*10**-3
  epsilon = sys.float_info.epsilon
  normal = 'fn'
  w_width = 100
  [lighting_x, lighting_y] = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
  lighting_x = np.concatenate(lighting_x)
  lighting_y = np.concatenate(lighting_y)
  lighting = np.vstack((lighting_x, lighting_y, np.zeros_like(lighting_x))).T
 
  measurement_num = lighting.shape[0] 
  sensor = np.array([0, 0, 0.0])
  sensor = np.tile(sensor, (measurement_num,1))
  sensor_normal = np.array([0, 0, 1])
  sensor_normal = np.tile(sensor_normal, (measurement_num,1))
  
  lighting_normal = np.array([0, 0, 1])
  lighting_normal = np.tile(lighting_normal, (measurement_num,1))

class MESH:
  v = igl.eigen.MatrixXd()
  f = igl.eigen.MatrixXi()
  fn = igl.eigen.MatrixXd()

parser = argparse.ArgumentParser()
parser.add_argument('lr', type=float, default=0.001)
args = parser.parse_args()

#device = torch.device('cuda:1')
device = torch.device('cpu')
folder_name = os.getcwd() + '/progress10-%f/'%args.lr
if not os.path.isdir(folder_name):
  os.mkdir(folder_name)


filename = os.getcwd() + '/setup.mat'
setup = scipy.io.loadmat(filename)

gt_transient = setup['gt_transient']
#mesh_location = '../mesh_processing/data/bunny.obj'
gt_mesh = MESH()
gt_mesh.v = igl.eigen.MatrixXd(torch.from_numpy(setup['gt_v']).numpy())
gt_mesh.f = igl.eigen.MatrixXi(torch.from_numpy(setup['gt_f']).numpy())
igl.per_face_normals(gt_mesh.v, gt_mesh.f, gt_mesh.fn)

opt = OPT(5000)
render_opt = OPT(50000)

space_carving_location = os.getcwd() + '/space_carving_mesh.obj'
space_carving_mesh = MESH()
igl.readOBJ(space_carving_location, space_carving_mesh.v, space_carving_mesh.f)

mesh = MESH()
#mesh.v = space_carving_mesh.v
#mesh.f = space_carving_mesh.f
mesh.v = np.array(gt_mesh.v)
mesh.v[:,2] += 0.2
#mesh.v += np.random.normal(0,0.1,mesh.v.shape)
mesh.v = igl.eigen.MatrixXd(mesh.v)

mesh.f = gt_mesh.f
igl.per_face_normals(mesh.v, mesh.f, mesh.fn)

mesh_optimization = MESH()
mesh_optimization.v = torch.from_numpy(np.array(mesh.v)).to(device)
mesh_optimization.v.requires_grad_()
mesh_optimization.f = torch.from_numpy(np.array(mesh.f)).long().to(device)

l2, transient = rendering.evaluate_L2(gt_transient, mesh, render_opt)
print('%05d update time: %5.5f L2 loss: %5.5f '% (0, 0, l2))

filename = folder_name + 'init.mat'
scipy.io.savemat(filename, mdict={'f': np.array(mesh.f), 'v':np.array(mesh.v), 'optim_v':mesh_optimization.v.cpu().data.numpy(), 'gt_v':np.array(gt_mesh.v), 'transient':transient, 'l2': l2, 'gt_transient': gt_transient})

optimizer = optim.Adam([mesh_optimization.v], lr = args.lr)

T = 200

l2_record = np.empty(T*opt.lighting.shape[0])	

for t in range(T):
    if t%3 == 0:
        opt.w_width -= 1
    
    for i, index in enumerate(np.random.permutation(opt.lighting.shape[0])):
        tic = time.time()
        optimizer.zero_grad()

        loss = rendering.loss_fun_weighted_time(index, gt_transient, mesh, mesh_optimization, opt, render_opt, device)
        loss.backward()
        optimizer.step()
        #print(mesh_optimization.v.grad)

        mesh.v = igl.eigen.MatrixXd(mesh_optimization.v.cpu().data.numpy())
        S = igl.eigen.MatrixXd()
        I = igl.eigen.MatrixXi()
        C = igl.eigen.MatrixXd()
        N = igl.eigen.MatrixXd()
        igl.signed_distance(mesh.v, space_carving_mesh.v, space_carving_mesh.f, igl.SignedDistanceType(0), S,I,C,N)   
        for x in list(compress(range(mesh.v.rows()), S>0)):
            mesh.v.setRow(x, C.row(x))

        igl.per_face_normals(mesh.v, mesh.f, mesh.fn)
        
        l2, transient = rendering.evaluate_L2(gt_transient, mesh, render_opt)
        print('%05d update time: %5.5f L2 loss: %5.5f '% (t*opt.lighting.shape[0]+i, time.time() - tic, l2 ))
        l2_record[t*opt.lighting.shape[0]+i] = l2	
        filename = folder_name + '%05d.mat'%(t*opt.lighting.shape[0]+i)
        scipy.io.savemat(filename, mdict={'v':np.array(mesh.v),  'transient':transient, 'l2':l2, 'origin_v': mesh_optimization.v.data.cpu().numpy(), 'grad': mesh_optimization.v.grad.data.cpu().numpy(), 'w_width': opt.w_width})
        #print(mesh_optimization.v.grad.data)
        mesh_optimization.v.data = torch.from_numpy(np.array(mesh.v)).to(device)

filename = folder_name + 'loss_val.mat'
scipy.io.savemat(filename, mdict={'l2':l2_record})
