clear; clc; close all;

addpath(genpath('../../../gptoolbox/'));

addpath('../../MarchingCubes/');
load('setup');

if ~exist('lighting', 'var')
  [lighting_x, lighting_y] = meshgrid(-1:1:1, -1:1:1);
  lighting = [lighting_x(:) lighting_y(:) zeros(size(gt_transient,1),1)];
  sensor = repmat([0,0,0], size(gt_transient,1), 1);
  bin_width = 5*10^-3;
end

threshold = 5*bin_width;

interval = threshold/2;
[space_carving_X, space_carving_Y, space_carving_Z] = meshgrid(-2:interval:2,-2:interval:2,0.8:interval:1.3);
occupancy = ones(size(space_carving_X));
occupancy_new = ones(size(space_carving_X));


for i = 1:size(gt_transient,1)
    first_photon = find(gt_transient(i,:)~=0,1);
    total_distance = first_photon*5*10^-3;
    d1 = sqrt((space_carving_X-lighting(i,1)).^2 +  (space_carving_Y-lighting(i,2)).^2 + (space_carving_Z-lighting(i,3)).^2);
    d2 = sqrt((space_carving_X-sensor(i,1)).^2 +  (space_carving_Y-sensor(i,2)).^2 + (space_carving_Z-sensor(i,3)).^2);
    d = d1+d2;
    occupancy = d > (total_distance - threshold);
    occupancy_new = occupancy_new.*occupancy;
end

idx = (occupancy_new == 1);
figure;
plot3(space_carving_X(idx), space_carving_Y(idx), space_carving_Z(idx), 'r.');


[F,V] = MarchingCubes(space_carving_X,space_carving_Y,space_carving_Z,occupancy, 0.99);
figure;
plot3(V(:,1), V(:,2), V(:,3), 'r.');


%N = per_vertex_normals(V,F);
writeOBJ('space_carving_mesh.obj', V, F);
%save('space_carving_mesh', 'V', 'F', 'N');
