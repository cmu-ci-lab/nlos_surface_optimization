function compute_space_carving_mesh(transient, output_file)

addpath(genpath('../../../../gptoolbox/'));
addpath('../../MarchingCubes/');
setup = load(transient);

threshold = 10*setup.bin_width;

interval_x = .5/64;
interval_z = threshold/2;
[space_carving_X, space_carving_Y, space_carving_Z] = meshgrid(-.3:interval_x:.3,-.3:interval_x:.3,0:interval_z:setup.bin_width*size(setup.gt_transient,2)/2);
occupancy = ones(size(space_carving_X));
occupancy_new = ones(size(space_carving_X));


for i = 1:size(setup.gt_transient,1)
    i
    first_photon = find(setup.gt_transient(i,:)~=0,1);
    total_distance = first_photon*setup.bin_width;
    d1 = sqrt((space_carving_X-setup.lighting(i,1)).^2 +  (space_carving_Y-setup.lighting(i,2)).^2 + (space_carving_Z-setup.lighting(i,3)).^2);
    %d2 = sqrt((space_carving_X-sensor(i,1)).^2 +  (space_carving_Y-sensor(i,2)).^2 + (space_carving_Z-sensor(i,3)).^2);
    d = 2*d1;%+d2;
    occupancy = d > (total_distance - threshold);
    occupancy_new = occupancy_new.*occupancy;
end

[F,V] = MarchingCubes(space_carving_X,space_carving_Y,space_carving_Z,occupancy_new, 0.1);




figure;
plot3(V(:,1), V(:,2), V(:,3), 'r.');


writeOBJ(output_file, V, F);

hold on;
plot3(setup.gt_v(:,1),setup.gt_v(:,2),setup.gt_v(:,3),'b.');
