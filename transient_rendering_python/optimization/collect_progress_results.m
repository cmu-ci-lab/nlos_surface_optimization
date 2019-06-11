clear; clc; close all;

folder = 'progress/';

output_folder = 'result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
output_file = [output_folder 'info.mat'];

file = dir([folder '*.mat']);

T = size(file,1) - 2 ;

transient_collection = cell(T,1);
origin_v_collection = cell(T,1);
v_collection = cell(T,1);
grad_collection = cell(T,1);
for t = 1:T,
    filename = [folder num2str(t-1, '%05d')];
    load(filename);
    transient_collection{t,1} = transient;
    origin_v_collection{t,1} = origin_v;
    v_collection{t,1} = v;
    grad_collection{t,1} = grad;
end

load([folder 'init.mat']);
transient_collection = [transient; transient_collection];
origin_v_collection = [optim_v; origin_v_collection];
v_collection = [v; v_collection];

load([folder 'loss_val']);
save(output_file, 'transient_collection', 'grad_collection', 'origin_v_collection', 'v_collection', 'gt_transient', 'gt_v', 'f', 'l2');


