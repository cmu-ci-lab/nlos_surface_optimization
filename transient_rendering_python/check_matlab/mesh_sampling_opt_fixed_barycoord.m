function transient = mesh_sampling_opt_fixed_barycoord(mesh, lighting, sensor, lighting_normal, sensor_normal, barycoord, opt)

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

transient = zeros(1,max_distance_bin);
v2 = nan(sample_num,3);
d1 = nan(sample_num,1);
d2 = nan(sample_num,1);
cos_theta1 = nan(sample_num,1);
cos_theta2 = nan(sample_num,1);
normalMap = zeros(sample_num,3);

sampled_face = barycoord(:,1) + 1;
sampled_point =  bsxfun(@times,1-barycoord(:,2)-barycoord(:,3),mesh.v(mesh.f(sampled_face,1),:)) +  ...
  bsxfun(@times,barycoord(:,2),mesh.v(mesh.f(sampled_face,2),:)) +  ...
  bsxfun(@times,barycoord(:,3),mesh.v(mesh.f(sampled_face,3),:)) ;
  
  
mesh_area = sum(doublearea(mesh.v, mesh.f))/2;

% visibility check to the lighting
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

if normal == 'n'
    normalMap(inds,:) =  bsxfun(@times,barycentric_coord(inds,1),mesh.n(mesh.f(sampled_face(inds),1),:)) + ...
        bsxfun(@times,barycentric_coord(inds,2),mesh.n(mesh.f(sampled_face(inds),2),:)) +  ...
        bsxfun(@times,barycentric_coord(inds,s3),mesh.n(mesh.f(sampled_face(inds),3),:));
    normalMap = normalMap ./ repmat(sqrt(sum(normalMap .^ 2, 2)), [1 3]);
else
    normalMap(inds,:) = mesh.fn(sampled_face(inds),:);
end


distance_bin = nan(sample_num,1);
intensity = zeros(sample_num,1);



cos_theta1(inds) = max(0,dot(normalMap(inds, :),v1(inds,:),2));
cos_theta2(inds) = max(0,dot(normalMap(inds, :),v2(inds,:),2));

distance_bin(inds) = ceil((d1(inds)+d2(inds))/distance_resolution);
inds = distance_bin <= max_distance_bin;
intensity(inds) = cos_theta1(inds).*cos_theta2(inds)./(d1(inds).^2)./(d2(inds).^2);



u = unique(distance_bin(inds));
for j = 1:length(u)
   transient(u(j)) = transient(u(j)) + sum(intensity(distance_bin == u(j)));
end

transient = transient * mesh_area / opt.sample_num;
 
end
