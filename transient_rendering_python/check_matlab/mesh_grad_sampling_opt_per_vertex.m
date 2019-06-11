function grad = mesh_grad_sampling_opt_per_vertex(mesh, lighting, sensor, lighting_normal, sensor_normal, opt)

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

[sampled_point,sampled_face,barycentric_coord,~,mesh_area] = random_points_on_mesh_w_mesh_area(mesh.v, mesh.f, opt.sample_num);


grad = zeros(opt.max_distance_bin,3*size(mesh.v,1));

for vertex = 1:size(mesh.v,1)
    related_face = find(sum(mesh.f == vertex, 2) > 0);
    
    for f = 1:length(related_face)
        face = mesh.f(related_face(f),:);
        vertex_idx = find(face == vertex);
        
        ind = (sampled_face == related_face(f));
        sample_num = sum(ind);
        phi = barycentric_coord(ind,1);
        u = barycentric_coord(ind,2);
        v = barycentric_coord(ind,3);
        
        sampled_point = bsxfun(@times, mesh.v(face(1),:), phi) + bsxfun(@times, mesh.v(face(2),:), u) + bsxfun(@times, mesh.v(face(3),:), v);
        
        
        v2 = nan(sample_num,3);
        d2 = nan(sample_num,1);
        cos_theta1 = nan(sample_num,1);
        cos_theta2 = nan(sample_num,1);
        
        
        v1 = bsxfun(@minus, lighting, sampled_point);
        d1 = sqrt(v1(:,1).^2 + v1(:,2).^2 + v1(:,3).^2);
        v1 = bsxfun(@rdivide, v1, d1);
        
        
        [barycoord] = intersectRayTriangleBatchRays_mex(repmat(lighting, sample_num, 1), -v1, mesh.v, mesh.f);
        
        inds = find(barycoord(:,1) == related_face(f));
        
        v2(inds,:) = bsxfun(@minus, sensor, sampled_point(inds,:));
        d2(inds) = sqrt(v2(inds,1).^2 + v2(inds,2).^2 + v2(inds,3).^2);
        v2(inds,:) = bsxfun(@rdivide, v2(inds,:), d2(inds));
        
        [barycoord] = intersectRayTriangleBatchRays_mex(repmat(sensor, length(inds), 1), -v2(inds,:), mesh.v, mesh.f);
        
        new_inds = barycoord(:,1) == related_face(f);
        
        % index that a point on mesh is visible to both lighting and sensor
        inds = inds(new_inds);
        
        normalMap = mesh.fn(related_face(f),:);
        
        distance_bin = nan(sample_num,1);
        intensity = nan(sample_num,1);
        
        cos_theta1(inds) = max(0,v1(inds,:)*normalMap');
        cos_theta2(inds) = max(0,v2(inds,:)*normalMap');
        
        distance_bin(inds) = ceil((d1(inds)+d2(inds))/distance_resolution);
        inds = distance_bin <= max_distance_bin;
        intensity(inds) = cos_theta1(inds).*cos_theta2(inds)./(d1(inds).^2)./(d2(inds).^2);
        
        
        gx1 = nan(sample_num,3);
        gx2 = nan(sample_num,3);
        t1 = nan(sample_num,3);
        t2 = nan(sample_num,3);
        
        gn_tmp = nan(sample_num,3);
        gn = nan(sample_num,3);
        
        cos_tmp = nan(sample_num,1);
        g = nan(sample_num,3);
        
        gx1(inds,:) = bsxfun(@times, normalMap, d2(inds).*cos_theta2(inds) + d1(inds).*cos_theta1(inds));
        gx2(inds,:) = bsxfun(@times, bsxfun(@times, v1(inds,:), d2(inds)) + bsxfun(@times, v2(inds,:), d1(inds)), cos_theta1(inds).*cos_theta2(inds));
        
        
        t1(inds,:) = bsxfun(@rdivide, -gx1(inds,:) + 3*gx2(inds,:), d1(inds).^3.*d2(inds).^3);
        t2(inds,:) = bsxfun(@times, normalMap, intensity(inds));
        
        gn_tmp(inds,:) = bsxfun(@rdivide, bsxfun(@times, v1(inds,:), cos_theta2(inds)) + bsxfun(@times, v2(inds,:), cos_theta1(inds)),(d1(inds).^2.*d2(inds).^2));
        cos_tmp(inds) = gn_tmp(inds,:)*normalMap';
        
        gn(inds,:) = gn_tmp(inds,:) - bsxfun(@times, cos_tmp(inds), normalMap);
        
        t2(inds,:) = (t2(inds,:) + gn(inds,:))/ (2*mesh_area(related_face(f)));
        
        if vertex_idx == 1
            e = mesh.v(face(3),:) -  mesh.v(face(2),:);
            e_mat = [0 e(3) -e(2); -e(3) 0 e(1); e(2) -e(1) 0];
            g(inds,:) = bsxfun(@times, t1(inds,:), phi(inds)) - t2(inds,:)*e_mat;
        elseif vertex_idx == 2
            e = mesh.v(face(1),:) -  mesh.v(face(3),:);
            e_mat = [0 e(3) -e(2); -e(3) 0 e(1); e(2) -e(1) 0];
            g(inds,:) = bsxfun(@times, t1(inds,:), u(inds)) - t2(inds,:)*e_mat;
        else
            e = mesh.v(face(2),:) -  mesh.v(face(1),:);
            e_mat = [0 e(3) -e(2); -e(3) 0 e(1); e(2) -e(1) 0];
            g(inds,:) = bsxfun(@times, t1(inds,:), v(inds)) - t2(inds,:)*e_mat;
        end
        
        
        u = unique(distance_bin(inds));
        for j = 1:length(u)
            grad(u(j),3*vertex-2:3*vertex) = grad(u(j),3*vertex-2:3*vertex) + sum(g(distance_bin == u(j),:),1)*mesh_area(related_face(f));
        end
        
    end
    
end
end
