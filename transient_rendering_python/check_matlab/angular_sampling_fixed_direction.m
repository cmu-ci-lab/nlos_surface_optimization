function [transient, grad_total] = angular_sampling_fixed_direction(v, f, lighting, sensor, lighting_normal, sensing_normal, direction, opt)

num_v = length(v)/3;
mesh.v = reshape(v, num_v, 3);
mesh.f = f;
mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);


triangleVertex1 = mesh.v(mesh.f(:, 1), :);
triangleVertex2 = mesh.v(mesh.f(:, 2), :);
triangleVertex3 = mesh.v(mesh.f(:, 3), :);

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


if isfield(opt, 'epsilon')
    epsilon = opt.epsilon;
else
    epsilon = distance_resolution/2;
end


if isfield(opt, 'normal')
    normal = opt.normal;
else
    normal = 'n';
end


transient = zeros(1,max_distance_bin);


% figure; hold on;
% plot3(mesh.v(:,1), mesh.v(:,2), mesh.v(:,3), 'k.');
% plot3(sensor(1), sensor(2), sensor(3), 'r.');
% plot3(lighting(1), lighting(2), lighting(3), 'r.');

%phi = 2*pi*rand(sample_num,1);
%theta = acos(rand(sample_num,1));

%R = vrrotvec2mat(vrrotvec([0 0 1],lighting_normal));

%direction = [sin(theta).*cos(phi) sin(theta).*sin(phi) cos(theta)]*R';

%[intersect, t, u, v] = intersectRayTriangleBatchDirections(lighting', direction, triangleVertex1, triangleVertex2, triangleVertex3);
[intersect, t, u, v] = intersectRayTriangleBatchDirectionsOpt(lighting, direction, triangleVertex1, triangleVertex2, triangleVertex3);

d1 = nan(sample_num,1);
d2 = nan(sample_num,1);
distance_bin = nan(sample_num,1);
intensity = nan(sample_num,1);

uMap = nan(sample_num,1);
vMap = nan(sample_num,1);
triangleIndexMap = nan(sample_num,1);

for i = 1:sample_num
    intersectInds = find(intersect(:,i));
    if isempty(intersectInds)
        uMap(i) = NaN;
        vMap(i) = NaN;
        triangleIndexMap(i) = NaN;
    else
        if (numel(intersectInds) == 1),
            d1(i) = abs(t(intersectInds,i));
            triangleIndex = intersectInds;
        else
            [d1(i), nearestIntersection] = min(abs(t(intersectInds,i)));
            triangleIndex = intersectInds(nearestIntersection);
        end;
        uMap(i) = u(triangleIndex,i);
        vMap(i) = v(triangleIndex,i);
        triangleIndexMap(i) = triangleIndex;
    end
end

intersection_p = nan(sample_num,3);
normalMap = zeros(sample_num, 3);
v2 = nan(sample_num,3);
cos_theta2 = nan(sample_num,1);

inds = find(triangleIndexMap > 0);


intersection_p(inds,:) =  bsxfun(@times, 1 - uMap(inds) - vMap(inds),  mesh.v(mesh.f(triangleIndexMap(inds), 3), :)) + ...
    bsxfun(@times, uMap(inds) ,  mesh.v(mesh.f(triangleIndexMap(inds), 1), :)) + ...
    bsxfun(@times, vMap(inds),  mesh.v(mesh.f(triangleIndexMap(inds), 2), :)) ;


%plot3(intersection_p(:,1), intersection_p(:,2), intersection_p(:,3), 'g.');



% v2(inds,1) = (sensor(1) - intersection_p(inds,1))./d2(inds);
% v2(inds,2) = (sensor(2) - intersection_p(inds,2))./d2(inds);
% v2(inds,3) = (sensor(3) - intersection_p(inds,3))./d2(inds);
v2(inds,:) = bsxfun(@minus, sensor, intersection_p(inds,:));




% visibility check
% [intersect, t, ~, ~] = intersectRayTriangleBatchRays(intersection_p(inds, :), v2(inds,:), triangleVertex1, triangleVertex2, triangleVertex3);
% intersect(t<-eps) = 0;
%


d2(inds) = sqrt(v2(inds,1).^2 + v2(inds,2).^2 + v2(inds,3).^2);
v2(inds,:) = bsxfun(@rdivide, v2(inds,:), d2(inds));
[intersect, t, ~, ~] = intersectRayTriangleBatchDirectionsOpt(sensor, -v2(inds,:), triangleVertex1, triangleVertex2, triangleVertex3);
intersect(t<=0) = 0;
intersect(bsxfun(@gt, t, d2(inds)'+epsilon)) = 0;


triangleIndexMap(inds(sum(intersect)>1)) = 0;

inds = find(triangleIndexMap > 0);

%plot3(intersection_p(inds,1), intersection_p(inds,2), intersection_p(inds,3), 'g.');


%
% for i = 1:size(intersect,2)
%     intersectInds = find(intersect(:,i));
%     if (numel(intersectInds) ~= 1),
%         closest_dist = min(t(intersectInds,i));
%         if closest_dist < d2(inds(i))
%             triangleIndexMap(inds(i)) = 0;
%         end
%     end;
% end
%
% inds = find(triangleIndexMap > 0);



if normal == 'n'
    normalMap(inds,:) =  bsxfun(@times, 1 - uMap(inds) - vMap(inds),  mesh.n(mesh.f(triangleIndexMap(inds), 3), :)) + ...
        bsxfun(@times, uMap(inds) ,  mesh.n(mesh.f(triangleIndexMap(inds), 1), :)) + ...
        bsxfun(@times, vMap(inds),  mesh.n(mesh.f(triangleIndexMap(inds), 2), :)) ;
    
    normalMap = normalMap ./ repmat(sqrt(sum(normalMap .^ 2, 2)), [1 3]);
else
    normalMap(inds,:) = mesh.fn(triangleIndexMap(inds),:);
end


cos_theta2(inds) = max(0,dot(normalMap(inds, :),v2(inds,:),2));

distance_bin(inds) = ceil((d1(inds)+d2(inds))/distance_resolution);
inds = distance_bin <= max_distance_bin;
intensity(inds) = cos_theta2(inds)./(d2(inds).^2);

u = unique(distance_bin(inds));
for j = 1:length(u)
    transient(u(j)) = transient(u(j)) + sum(intensity(distance_bin == u(j)));
end

transient = transient * 2 * pi /size(direction,1);
transient = sum(transient);


if nargout == 1
    return
end

[grad_total] = find_gradient_fixed_direction(mesh, lighting, sensor, lighting_normal, sensing_normal, direction);
grad_total = sum(grad_total,1)';



end