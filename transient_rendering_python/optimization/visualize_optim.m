clear; clc; close all;

% load('setup');
% figure; 
% for i = 1:size(gt_transient,1)
%     plot(gt_transient(i,:));
%     pause(0.5);
% end

interval = 30;
folder = 'result/';
load([folder 'info-0.000010.mat']);

figure;
plot(l2);
title('loss function');

if ~exist('gt_f', 'var')
    gt_f = [0,2,1;0, 3,2; 4,3,0;1,2,5] +1;
end
f = f + 1;

[x,y] = meshgrid(-1:1, -1:1);

figure; hold on;
plot3(x(:), y(:), zeros(length(x(:)),1), 'r.');

for i = 1:size(gt_f,1)
    h = fill3(gt_v(gt_f(i,:),1),gt_v(gt_f(i,:),2),gt_v(gt_f(i,:),3), 'r');
    alpha(h,0.5);
end



T = size(origin_v_collection,1);

v2 = nan(T,1);

for t = 1:interval:T,
 
    v = v_collection{t,1};
    v2(t) = norm(gt_v-v,'fro');
   
    for i = 1:size(f,1)
        h(i) = fill3(v(f(i,:),1), v(f(i,:),2), v(f(i,:),3), 'b');
    end

    drawnow;
    pause(0.3);


    for i = 1:size(f,1)
        delete(h(i));
    end
end

figure; 
plot(v2);
title('vertex distance');

figure;
plot(gt_transient(1,:), 'r');
hold on;
for t = 1:interval:T,
    title(num2str(t));
    transient = transient_collection{t,1};
    h1 = plot(transient(1,:));

    drawnow;
    pause(0.5);
    delete(h1);
end


