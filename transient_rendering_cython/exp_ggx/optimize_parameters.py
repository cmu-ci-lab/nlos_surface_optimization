import numpy as np
import sys, os
import scipy.io
sys.path.append('../exp_bunny')

import time
import torch
from torch import optim
from adam_modified import Adam_Modified

import rendering


def optimize_albedo(mesh, gt_transient, weight, opt, T, global_counter, folder):
  optimization_albedo = torch.Tensor([mesh.albedo])
  optimization_albedo.requires_grad_()
  optimizer_albedo = optim.Adam([optimization_albedo], lr = opt.albedo_lr) 

  dummy_loss2 = optimization_albedo**2
  dummy_loss2.backward()
  l2_record = np.empty(opt.T)	
  for t in range(T):
    tic = time.time()
    transient, grad_albedo = rendering.inverseRenderingAlbedo(mesh, gt_transient, weight, opt)
    l2,  original_l2 = rendering.evaluate_loss_with_normal_smoothness(gt_transient, weight, transient, 0, mesh, opt)  
    print('%05d update time: %8.8f L2 loss: %8.8f albedo: %f'% (global_counter, time.time() - tic, l2, mesh.albedo))
    #filename = folder + 'test_%05d.mat'%(global_counter)
    #scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'albedo': mesh.albedo})

    global_counter += 1

    l2_record[t] = l2

    if t > 2:
      if (l2_record[t-1] - original_l2)/l2_record[t-1] < opt.loss_epsilon:
        break

    optimization_albedo.grad.data = torch.Tensor([grad_albedo])
    optimizer_albedo.step()
    mesh.albedo = optimization_albedo.data.numpy()[0]



  return global_counter, l2

def initial_fitting_albedo(mesh, gt_transient, weight, opt):
  transient, path = rendering.forwardRendering(mesh, opt)
  return np.sum(gt_transient * transient) /np.linalg.norm(transient)**2

def optimize_alpha(mesh, gt_transient, weight, opt, T, global_counter, folder):
  optimization_alpha = torch.Tensor([mesh.alpha])
  optimization_alpha.requires_grad_()
  optimizer_alpha = optim.Adam([optimization_alpha], lr = opt.alpha_lr) 

  dummy_loss2 = optimization_alpha**2
  dummy_loss2.backward()
  l2_record = np.empty(opt.T)	
  for t in range(T):
    tic = time.time()
    transient, grad_alpha = rendering.inverseRenderingAlpha(mesh, gt_transient, weight, opt)
    l2,  original_l2 = rendering.evaluate_loss_with_normal_smoothness(gt_transient, weight, transient, 0, mesh, opt)  
    print('%05d update time: %8.8f L2 loss: %8.8f alpha: %f'% (global_counter, time.time() - tic, l2, mesh.alpha))
    #filename = folder + 'test_%05d.mat'%(global_counter)
    #scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f, 'alpha': mesh.alpha})
    global_counter += 1

    l2_record[t] = l2

    if t > 2:
      if (l2_record[t-1] - original_l2)/l2_record[t-1] < opt.loss_epsilon:
        break

    optimization_alpha.grad.data = torch.Tensor([grad_alpha])
    optimizer_alpha.step()
    mesh.alpha = optimization_alpha.data.numpy()[0]

  return global_counter, l2

def optimize_shape(mesh, gt_transient, weight, opt, T, lr, gt_mesh, global_counter, folder):
  optimization_v = torch.from_numpy(mesh.v[mesh.v_edge==0,:])
  optimization_v.requires_grad_()
  optimization_v_edge = torch.from_numpy(mesh.v[mesh.v_edge==1,:])
  optimization_v_edge.requires_grad_()

  optimizer = Adam_Modified([{'params':optimization_v}, {'params': optimization_v_edge, 'lr': lr*opt.edge_lr_ratio}], lr = lr)
  dummy_loss = torch.sum(optimization_v) + torch.sum(optimization_v_edge)
  dummy_loss.backward()

  l2_record = np.empty(opt.T)	
  v2_record = np.empty(opt.T)	
  l2_original_record = np.empty(opt.T)	

  for t in range(T):
    tic = time.time()
    transient, grad, length = rendering.inverseRendering(mesh, gt_transient, weight, opt)	  
    smoothing_val, smoothing_grad = rendering.renderStreamedNormalSmoothing(mesh)
    l2,  original_l2 = rendering.evaluate_loss_with_normal_smoothness(gt_transient, weight, transient, smoothing_val, mesh, opt)  
    if t == 0:
      opt.smooth_weight = original_l2/smoothing_val/opt.smooth_ratio
      print('smoothness weight %f'%opt.smooth_weight)

    grad += opt.smooth_weight * smoothing_grad
    if opt.gt_mesh:
      v2 = rendering.compute_v2(mesh.v, gt_mesh)
    else:
      v2 = 0
    #filename = folder + 'test_%05d.mat'%(global_counter)
    #scipy.io.savemat(filename, mdict={'v':mesh.v, 'f':mesh.f})
    print('%05d update time: %8.8f L2 loss: %8.8f  old_l2 loss: %8.8f v2: %8.8f'% (global_counter, time.time() - tic, l2, original_l2, v2))
    global_counter += 1


    l2_record[t] = l2	
    l2_original_record[t] = original_l2
    v2_record[t] = v2

    if t > 2:
      if (l2_original_record[t-1] - original_l2)/l2_original_record[t-1] < opt.loss_epsilon:
        return global_counter, True, original_l2
      if (l2_record[t-1] - l2)/l2_record[t-1] < opt.loss_epsilon:
        return global_counter, True, original_l2


    optimization_v.grad.data = torch.from_numpy(grad[mesh.v_edge==0,:]).float()
    optimization_v_edge.grad.data = torch.from_numpy(grad[mesh.v_edge==1,:]).float()
    optimizer.step()
    mesh.v[mesh.v_edge==0,:] = np.array(optimization_v.data.numpy(), dtype=np.float32, order='C')
    mesh.v[mesh.v_edge==1,:] = np.array(optimization_v_edge.data.numpy(), dtype=np.float32, order='C')

  return global_counter, False, original_l2
