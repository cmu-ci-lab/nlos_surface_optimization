clear; clc; close all;

test_num = 1;

for i = 1:3
	lr = num2str(10^-i, '%f');

	folder = ['progress' num2str(test_num) '-'  lr  '/'];

	output_folder = ['result/info' num2str(test_num) '/'];
	if ~exist(output_folder, 'dir')
	    mkdir(output_folder);
	end
	output_file = [output_folder lr '.mat'];

	file = dir([folder '*.mat']);
        interval = 50;
	T = ceil((size(file,1) - 2)/interval) ;

	transient_collection = cell(T,1);
	origin_v_collection = cell(T,1);
	v_collection = cell(T,1);
	grad_collection = cell(T,1);
        l2_collection = nan(T,1);
	cnt = 1;
        for t = 1:interval:size(file,1)-2,
	    filename = [folder num2str(t-1, '%05d')];
	    load(filename);
	    transient_collection{cnt,1} = transient;
	    origin_v_collection{cnt,1} = origin_v;
	    v_collection{cnt,1} = v;
	    grad_collection{cnt,1} = grad;
            w_width_collection{cnt,1} = w_width;
	    l2_collection(cnt,1) = l2;
            cnt = cnt + 1;
        end

	load([folder 'init.mat']);
	transient_collection = [transient; transient_collection];
	origin_v_collection = [optim_v; origin_v_collection];
	v_collection = [v; v_collection];

	load([folder 'loss_val']);
	%save(output_file, 'transient_collection', 'grad_collection', 'origin_v_collection', 'v_collection', 'gt_transient', 'gt_v', 'f', 'l2_collection', 'w_width_collection');
	save(output_file, 'transient_collection', 'grad_collection', 'origin_v_collection', 'v_collection', 'gt_transient', 'gt_v', 'f', 'l2');

end
