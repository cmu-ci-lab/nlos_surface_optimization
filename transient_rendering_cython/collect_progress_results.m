clear; clc; close all;

test_num = 3;
interval = 10;
lr = num2str(10^-1, '%f');

%load(['setup/' num2str(test_num) '.mat']);
load(['setup/' 'plane.mat']);

folder = ['progress' num2str(test_num) '-'  lr  '/'];

output_folder = 'result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
output_file = [output_folder 'info' num2str(test_num) '-'  lr '.mat'];

file = dir([folder '*.mat']);

T = ceil((size(file,1) - 2)/interval) ;

transient_collection = cell(T,1);
%origin_v_collection = cell(T,1);
v_collection = cell(T,1);
grad_collection = cell(T,1);
l2_collection = nan(T,1);
l2_original_collection = nan(T,1);
cnt = 1;
for t = 1:interval:size(file,1)-1,
    filename = [folder num2str(t-1, '%05d') '.mat'];
    result = load(filename);
    transient_collection{cnt,1} = result.transient;
    %origin_v_collection{cnt,1} = origin_v;
    v_collection{cnt,1} = result.v;
    grad_collection{cnt,1} = result.grad;
    w_width_collection{cnt,1} = result.w_width;
    l2_collection(cnt,1) = result.l2;
    l2_original_collection(cnt,1) = result.l2_original;
    cnt = cnt + 1;
end

f = result.f;
load([folder 'loss_val']);
save(output_file, 'transient_collection', 'grad_collection', 'v_collection', 'gt_transient', 'gt_v', 'gt_f', 'f', 'l2', 'w_width_collection', 'l2_collection', 'l2_original_collection');

