clear; clc; close all;

addpath(genpath('../../../gptoolbox/'));

addpath('../../MarchingCubes/');
load('setup');

if ~exist('lighting', 'var')
  fprintf('no lighting info\n');
  [lighting_x, lighting_y] = meshgrid(-1:1:1, -1:1:1);
  lighting = [lighting_x(:) lighting_y(:) zeros(size(gt_transient,1),1)];
  sensor = repmat([0,0,0], size(gt_transient,1), 1);
  bin_width = 5*10^-3;
end

threshold = 5*bin_width;

interval_x = threshold/2;
interval_z = threshold/2;
[space_carving_X, space_carving_Y, space_carving_Z] = meshgrid(-3:interval_x:3,-3:interval_x:3,0.8:interval_z:1.6);
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

idx = (occupancy_new ~= 1);
figure;
plot3(space_carving_X(idx), space_carving_Y(idx), space_carving_Z(idx), 'r.');


[F,V] = MarchingCubes(space_carving_X,space_carving_Y,space_carving_Z,occupancy_new, 0.1);
figure;
plot3(V(:,1), V(:,2), V(:,3), 'r.');

writeOBJ('space_carving_mesh.obj', V, F);
