clear; clc; close all;

addpath(genpath('../../../nlos_inverse_rendering/'));
addpath(genpath('../../../gptoolbox/'));
mesh_location = '../../mesh_processing/data/bunny.obj';
mesh = readMesh(mesh_location, 'C');
mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);

load('../python_test');

f = 0.5;
z = 1.8;
sensor = [f 0 z];
lighting = [-f 0 z];


sensor_normal = [0 0 -1];
lighting_normal = [0 0 -1];



opt.sample_num = size(direction,1);
opt.max_distance_bin = length(angular_transient);
opt.normal = 'fn';
tic;
transient_matlab = angular_sampling_fixed_direction(mesh, lighting, sensor, lighting_normal, sensor_normal, direction, opt);
toc;
figure;
plot(transient_matlab,'r-'); hold on;
plot(angular_transient, 'b-.');
norm(transient_matlab-angular_transient)



