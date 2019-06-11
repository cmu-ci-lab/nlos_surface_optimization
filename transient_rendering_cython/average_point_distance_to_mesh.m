function [v2] = average_point_distance_to_mesh(origin, mesh)

direction = zeros(size(origin));
direction(:,3) = 1;
[barycoord] = intersectRayTriangleBatchRays_mex(origin, direction, mesh.v, mesh.f);

point = nan(size(barycoord,1),3);
idx = find(barycoord(:,1)~= 0);
point(idx,:) = bsxfun(@times, barycoord(idx,2), mesh.v(mesh.f(barycoord(idx,1),2),:)) + ...
    bsxfun(@times, barycoord(idx,3), mesh.v(mesh.f(barycoord(idx,1),3),:)) + ...
    bsxfun(@times, (1 - barycoord(idx,2) - barycoord(idx,3)), mesh.v(mesh.f(barycoord(idx,1),1),:));

distance = abs(point(:,3) - origin(:,3));
v2 = mean(distance(~isnan(distance)));


end