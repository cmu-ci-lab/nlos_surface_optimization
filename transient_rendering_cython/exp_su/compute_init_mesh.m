clear; clc; close all;

addpath(genpath('../../../gptoolbox/'));
addpath('../exp_bunny/');

%z_range = [1300, 2048];
z_range = [0, 1300];
[x, y, depth, albedo, vol] = cnlos_reconstruction_su(7, z_range);

figure; imagesc(albedo);
threshold = 10*10^-7;
mask = albedo > threshold;


v = [x(:) y(:) depth(:)];
v = double(v);

figure;
imagesc(mask);

figure; hold on;
plot3(v(mask(:),1), v(mask(:),2), v(mask(:),3), 'b.');

xlabel('x'); ylabel('y'); zlabel('z');


v = v(mask(:),:);
[face] = create_face(mask);
figure;
triplot(face,v(:,1),v(:,2));
hold on;
plot(v(:,1), v(:,2), 'r.');

output_location = 'cnlos_s_threshold.obj';
writeOBJ(output_location, v, [face(:,1) face(:,3) face(:,2)]);

load('../cnlos/data_s_u');

transient = reshape(rect_data,64*64, 2048);
transient(:,1:600) = 0;

res = linspace(-width, width, size(rect_data,1));
[x,y] = ndgrid(res, res);
lighting = [x(:), y(:)];
lighting = [lighting zeros(size(lighting,1),1)];

save('transient', 'transient', 'lighting');

