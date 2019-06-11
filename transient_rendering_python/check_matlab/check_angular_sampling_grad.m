clear; clc; close all;
addpath(genpath('../../../nlos_inverse_rendering/'));
addpath(genpath('../../../gptoolbox/'));

load('../optimization/grad-progress2/setup');
load('../optimization/grad-progress2/0');


gt_mesh.v = v;
gt_mesh.f = double(f+1);


%f1 = 0.5;
%z = 0;
%sensor = [f1 0 z];
%lighting = [-f1 0 z];


sensor_normal = [0 0 1];
lighting_normal = [0 0 1];



opt.sample_num = size(direction,1);
opt.max_distance_bin = length(angular_transient);
opt.normal = 'fn';


[d, dy, dh] = checkgrad('angular_sampling_fixed_direction', gt_mesh.v(:), 1e-8, gt_mesh.f, lighting, sensor, lighting_normal, sensor_normal, direction, opt);
reshape(dy./dh, 6, 3)
reshape(dh./grad(:), 6,3)

%fprintf('Matlab forward');
%tic;
%transient_matlab = angular_sampling_fixed_direction(gt_mesh, lighting, sensor, lighting_normal, sensor_normal, direction, opt);
%toc;
%fprintf('Matlab backward');
%tic;
%[grad_total] = find_gradient_fixed_direction(gt_mesh, lighting, sensor, lighting_normal, sensor_normal, direction);
%toc;
%grad_total = repmat(angular_transient',1,size(grad_total,2)).*grad_total;
%grad_total = reshape(sum(grad_total,1), size(v,1), size(v,2));
%grad = reshape(grad, size(grad,1), size(grad,2)*size(grad,3));

