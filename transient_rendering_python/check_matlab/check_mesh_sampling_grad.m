clear; clc; close all;

addpath(genpath('../../../nlos_inverse_rendering/'));
addpath(genpath('../../../gptoolbox/'));

mesh.v = [-1 -1 1; 1 -1 0.9; 1 1 1.1; -1 1 0.95; -2 1 0.99; -2 -1 1.1];
%mesh.f = [3 2 1];
%mesh.f = [6 5 4];
mesh.f = [1 3 2; 1 4 3; 6 4 1; 6 5 4];

lighting = [0 0 0];
sensor = [0.1 0.1 0];

lighting_normal = [0 0 1];
sensor_normal = [0 0 1];

opt.sample_num = 10000;
opt.max_distance_bin = 1134;
opt.normal = 'fn';
opt.distance_resolution = 5*10^-3;

[sampled_point,sampled_face,barycentric_coord,~,~] = random_points_on_mesh_w_mesh_area(mesh.v, mesh.f, opt.sample_num);
%intensity = mesh_sampling_with_fixed_direction(mesh.v(:), mesh.f, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt);
% dh is finite difference
[d, dy, dh] = checkgrad('mesh_sampling_with_fixed_sampling', mesh.v(:), 1e-8, mesh.f, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt);
reshape(dy./dh, size(mesh.v,1),3 )
%[r] = checkgrad2(@mesh_sampling_with_fixed_sampling, mesh.v(:), {mesh.f, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt});

