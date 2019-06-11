clear; clc; close all;


addpath(genpath('../../../nlos_inverse_rendering/'));
addpath(genpath('../../../gptoolbox/'));
%mesh_location = '../../mesh_processing/data/bunny.obj';
%mesh = readMesh(mesh_location, 'C');

load('../optimization/grad-progress2/setup');


mesh.v = v;
mesh.f = double(f + 1);
mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);


lighting_normal = [0 0 1];
sensor_normal = [0 0 1];

opt.sample_num = 5000;
opt.max_distance_bin = 1134;
opt.normal = 'fn';
opt.distance_resolution = bin_width;
T = 50000;
%T = 1;
output_folder = 'mesh_grad_progress/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


mesh_grad_vertex = zeros(1,3*size(mesh.v,1));
for t = 1:T
    tic;
    tmp_grad = sum(mesh_grad_sampling_opt_per_vertex(mesh, lighting, sensor, lighting_normal, sensor_normal, opt),1);
    filename = [output_folder num2str(t, '%05d')];
    save(filename, 'tmp_grad');
    mesh_grad_vertex = mesh_grad_vertex + tmp_grad;    
    toc;
end
mesh_grad_vertex = mesh_grad_vertex/T;

filename = [output_folder 'mesh_grad'];
save(filename, 'mesh_grad_vertex');
return
% toc;
% mesh_grad = zeros(opt.max_distance_bin, 3*size(mesh.v,1));
% tic;
% for t = 1:T
%     mesh_grad = mesh_grad + mesh_grad_sampling_opt(mesh, lighting, sensor, lighting_normal, sensor_normal, opt);
% end
% mesh_grad = mesh_grad/T;
% toc;

pervertex = reshape(mesh_grad_vertex(bin_num,:),3,6)'
%all_faces = reshape(mesh_grad(bin_num,:),3,6)'
grad


pervertex = bsxfun(@rdivide, pervertex, sqrt(sum(pervertex.^2,2)));
%all_faces = bsxfun(@rdivide, all_faces, sqrt(sum(all_faces.^2,2)));
grad = bsxfun(@rdivide, grad, sqrt(sum(grad.^2,2)));
%dot(pervertex, all_faces, 2)
dot(pervertex, grad, 2)
%dot(all_faces, grad, 2) 

% pervertex is having 20*10000*O(facenum)*vertex_num
% meshgrad is having 20*10000 random mesh samlpling 2*10^5
% grad is having 57000*5000 random angular sampling  2.85*10^8
