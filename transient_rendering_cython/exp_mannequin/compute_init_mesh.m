clear; clc; close all;

addpath('../cnlos/');
addpath(genpath('../../../gptoolbox/'));
addpath('../exp_bunny/');
[x, y, depth, albedo, vol] = cnlos_reconstruction(5);

figure; imagesc(albedo);
threshold = 3*10^-7;
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

output_location = 'cnlos_mannequin_threshold.obj';
writeOBJ(output_location, v, [face(:,1) face(:,3) face(:,2)]);

load('../cnlos/data_mannequin');

transient = reshape(rect_data,64*64, 2048);
transient(:,1:600) = 0;

res = linspace(-width, width, size(rect_data,1));
[x,y] = ndgrid(res, res);
lighting = [x(:), y(:)];
lighting = [lighting zeros(size(lighting,1),1)];

save('transient', 'transient', 'lighting');

