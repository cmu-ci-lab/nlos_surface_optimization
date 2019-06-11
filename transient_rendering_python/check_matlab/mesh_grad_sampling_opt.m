function grad = mesh_grad_sampling_opt(mesh, lighting, sensor, lighting_normal, sensor_normal, opt)

if isfield(opt, 'distance_resolution')
    distance_resolution = opt.distance_resolution;
else
    distance_resolution = 5*10^-3;   
end

if isfield(opt, 'max_distance_bin')
    max_distance_bin = opt.max_distance_bin;
else
    max_distance_bin = 1000;
end

if isfield(opt, 'sample_num')
    sample_num = opt.sample_num;
else
    sample_num = 10000;
end


if isfield(opt, 'normal')
    normal = opt.normal;
else
    normal = 'n';
end


v2 = nan(sample_num,3);
d2 = nan(sample_num,1);
cos_theta1 = nan(sample_num,1);
cos_theta2 = nan(sample_num,1);
normalMap = zeros(sample_num,3);


[sampled_point,sampled_face,barycentric_coord,~,mesh_area] = random_points_on_mesh_w_mesh_area(mesh.v, mesh.f, sample_num);


v1 = bsxfun(@minus, lighting, sampled_point);
d1 = sqrt(v1(:,1).^2 + v1(:,2).^2 + v1(:,3).^2);
v1 = bsxfun(@rdivide, v1, d1);


[barycoord] = intersectRayTriangleBatchRays_mex(repmat(lighting, sample_num, 1), -v1, mesh.v, mesh.f);

inds = find(barycoord(:,1) == sampled_face);

v2(inds,:) = bsxfun(@minus, sensor, sampled_point(inds,:));
d2(inds) = sqrt(v2(inds,1).^2 + v2(inds,2).^2 + v2(inds,3).^2);
v2(inds,:) = bsxfun(@rdivide, v2(inds,:), d2(inds));

[barycoord] = intersectRayTriangleBatchRays_mex(repmat(sensor, length(inds), 1), -v2(inds,:), mesh.v, mesh.f);

new_inds = barycoord(:,1) == sampled_face(inds);

% index that a point on mesh is visible to both lighting and sensor
inds = inds(new_inds);

normalMap(inds,:) = mesh.fn(sampled_face(inds),:);

distance_bin = nan(sample_num,1);
intensity = nan(sample_num,1);



cos_theta1(inds) = max(0,dot(normalMap(inds, :),v1(inds,:),2));
cos_theta2(inds) = max(0,dot(normalMap(inds, :),v2(inds,:),2));

distance_bin(inds) = ceil((d1(inds)+d2(inds))/distance_resolution);
inds = distance_bin <= max_distance_bin;
intensity(inds) = cos_theta1(inds).*cos_theta2(inds)./(d1(inds).^2)./(d2(inds).^2);


gx1 = nan(sample_num,3);
gx2 = nan(sample_num,3);
t1 = nan(sample_num,3);
t2 = nan(sample_num,3);
t2_tmp = nan(sample_num,3);

gn_tmp = nan(sample_num,3);
gn = nan(sample_num,3);

cos_tmp = nan(sample_num,1);
e1 = nan(sample_num,3);
e2 = nan(sample_num,3);
e3 = nan(sample_num,3);
g1 = nan(sample_num,3);
g2 = nan(sample_num,3);
g3 = nan(sample_num,3);

gx1(inds,:) = bsxfun(@times, normalMap(inds,:), d2(inds).*cos_theta2(inds) + d1(inds).*cos_theta1(inds));
gx2(inds,:) = bsxfun(@times, bsxfun(@times, v1(inds,:), d2(inds)) + bsxfun(@times, v2(inds,:), d1(inds)), cos_theta1(inds).*cos_theta2(inds));


t1(inds,:) = bsxfun(@rdivide, -gx1(inds,:) + 3*gx2(inds,:), d1(inds).^3.*d2(inds).^3);
t2_tmp(inds,:) = bsxfun(@times, normalMap(inds,:), intensity(inds));

gn_tmp(inds,:) = bsxfun(@rdivide, bsxfun(@times, v1(inds,:), cos_theta2(inds)) + bsxfun(@times, v2(inds,:), cos_theta1(inds)),(d1(inds).^2.*d2(inds).^2));
cos_tmp(inds) = dot(gn_tmp(inds,:), normalMap(inds,:), 2);
gn(inds,:) = gn_tmp(inds,:) - bsxfun(@times, cos_tmp(inds), normalMap(inds,:));

t2_tmp(inds,:) = bsxfun(@rdivide, t2_tmp(inds,:) + gn(inds,:), 2*mesh_area(sampled_face(inds)));

e1(inds, :) = mesh.v(mesh.f(sampled_face(inds),3),:) -  mesh.v(mesh.f(sampled_face(inds),2),:);
e2(inds, :) = mesh.v(mesh.f(sampled_face(inds),1),:) -  mesh.v(mesh.f(sampled_face(inds),3),:);
e3(inds, :) = mesh.v(mesh.f(sampled_face(inds),2),:) -  mesh.v(mesh.f(sampled_face(inds),1),:);

g1(inds,:) = bsxfun(@times, t1(inds,:), barycentric_coord(inds,1)) + cross(t2_tmp(inds,:), e1(inds,:), 2);
g2(inds,:) = bsxfun(@times, t1(inds,:), barycentric_coord(inds,2)) + cross(t2_tmp(inds,:), e2(inds,:), 2);
g3(inds,:) = bsxfun(@times, t1(inds,:), barycentric_coord(inds,3)) + cross(t2_tmp(inds,:), e3(inds,:), 2);


grad = zeros(max_distance_bin, 3*size(mesh.v,1));
inds = find(inds);
for i = 1:length(inds)
    f = sampled_face(inds(i));
    
    grad(distance_bin(inds(i)),3*mesh.f(f,1)-2:3*mesh.f(f,1)) = grad(distance_bin(inds(i)),3*mesh.f(f,1)-2:3*mesh.f(f,1)) + g1(inds(i),:);
    grad(distance_bin(inds(i)),3*mesh.f(f,2)-2:3*mesh.f(f,2)) = grad(distance_bin(inds(i)),3*mesh.f(f,2)-2:3*mesh.f(f,2)) + g2(inds(i),:);
    grad(distance_bin(inds(i)),3*mesh.f(f,3)-2:3*mesh.f(f,3)) = grad(distance_bin(inds(i)),3*mesh.f(f,3)-2:3*mesh.f(f,3)) + g3(inds(i),:);

end

grad = grad*sum(mesh_area)/sample_num;

end