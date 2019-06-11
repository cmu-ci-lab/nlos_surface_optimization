clear; clc; close all;

test_num = 'b-3';
interval = 1;
lr = num2str(10, '%f');

load(['setup/bunny_transient.mat']);

folder = ['progress-' test_num '-'  lr  '/'];

output_folder = ['result-mesh-' num2str(test_num) '-' lr '/'];
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

file = dir([folder '*.mat']);

for t = 1:interval:size(file,1)-1,
    t
    filename = [folder num2str(t-1, '%05d') '.mat'];
    result = load(filename);
    output_filename = [output_folder num2str(t-1, '%05d') '.obj'];
    writeOBJ(output_filename, result.v, result.f+1);
end


