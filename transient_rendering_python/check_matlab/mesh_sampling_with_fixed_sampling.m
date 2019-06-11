function [intensity, dv] = mesh_sampling_with_fixed_sampling(v, f, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt)

v_num = length(v)/3;
mesh.v = reshape(v, v_num, 3);
mesh.f = f;
mesh.fn = normalizerow(normals(mesh.v,mesh.f)+eps);

transient = mesh_sampling_opt_fixed_sampling(mesh, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt);
intensity = sum(transient);

if nargout == 1
    return
end


dv = zeros(v_num,3);
for k = 1:v_num
    dv(k,:) = mesh_grad_sampling_opt_per_vertex_with_fixed_sampling(k, mesh, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord,sampled_face, opt);   
end
dv = dv(:);

%mesh_grad_vertex = mesh_grad_sampling_opt_with_fixed_sampling(mesh, lighting, sensor, lighting_normal, sensor_normal, barycentric_coord, sampled_face, opt);
%mesh_grad_vertex = sum(mesh_grad_vertex,1);
%mesh_grad_vertex = reshape(mesh_grad_vertex, 3, v_num)';
%dv = mesh_grad_vertex(:);
   
end

