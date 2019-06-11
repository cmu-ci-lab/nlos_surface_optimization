clear; clc; close all;

addpath(genpath('../../gptoolbox/'));

test_num = 1;

transient_location = ['setup/' num2str(test_num) '.mat'];
output_location = ['cnlos_mesh/' num2str(test_num) '.obj'];

[x, y, depth, albedo] = cnlos(transient_location);

v = [-x(:) y(:) depth(:)];
%threshold = 1.1*10^-9;
threshold = 0;
v(albedo(:) < threshold,:) = [];
v = double(v);

tri = delaunay(v(:,1), v(:,2));
figure;
triplot(tri,v(:,1),v(:,2));

writeOBJ(output_location, v, [tri(:,1) tri(:,3) tri(:,2)]);

figure; hold on;
plot3(v(:,1), v(:,2), v(:,3), 'r.');
xlabel('x'); ylabel('y'); zlabel('z');

[X,Y] = meshgrid(-1:0.2:1, -1:0.2:1);
Z = -0.5 + sqrt(4-X.^2-Y.^2);

surf(X,Y,Z);
