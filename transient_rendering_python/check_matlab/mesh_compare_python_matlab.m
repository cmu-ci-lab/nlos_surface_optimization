clear; clc; close all;

addpath(genpath('../../../nlos_inverse_rendering/'));
addpath(genpath('../../../gptoolbox/'));
mesh_location = '../../mesh_processing/data/bunny.obj';
%mesh = readMesh(mesh_location, 'C');
%mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);

load('../python_mesh_grad');



if exist('mesh_v', 'var')
  mesh.v = mesh_v;
  mesh.f = double(mesh_f) + 1;
  opt.sample_num = size(barycoord,1);
end
mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);


sensor_normal = [0 0 1];
lighting_normal = [0 0 1];


opt.max_distance_bin = length(mesh_transient);
opt.normal = 'fn';
tic;
transient = mesh_sampling_opt_fixed_barycoord(mesh, lighting, sensor, lighting_normal, sensor_normal, barycoord, opt);
toc;
figure;
plot(transient,'r-'); hold on;
plot(mesh_transient, 'b-.');
norm(transient-mesh_transient)



