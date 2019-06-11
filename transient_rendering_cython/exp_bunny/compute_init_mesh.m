function compute_init_mesh(transient_location, output_location, threshold)
addpath(genpath('../../../gptoolbox/'));

[x, y, depth, albedo, vol] = cnlos(transient_location);

figure; imagesc(albedo);
if ~exist('threshold','var')
    threshold = 0.8*10^-3;
end
mask = albedo > threshold;


v = [-x(:) y(:) depth(:)];
v = double(v);

figure;
imagesc(mask);

figure; hold on;
plot3(v(mask(:),1), v(mask(:),2), v(mask(:),3), 'b.');

xlabel('x'); ylabel('y'); zlabel('z');

load(transient_location);
hold on;
plot3(gt_v(:,1),gt_v(:,2), gt_v(:,3),'r.');


v = v(mask(:),:);
[face] = create_face(mask);
figure;
triplot(face,v(:,1),v(:,2));
hold on;
plot(v(:,1), v(:,2), 'r.');

writeOBJ(output_location, v, [face(:,1) face(:,2) face(:,3)]);

