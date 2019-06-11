function [grad_total] = find_gradient_fixed_direction(mesh, lighting, sensor, lighting_normal, sensor_normal, direction)



vertex_num = size(mesh.v,1);
face_num = size(mesh.f,1);
S1 = sparse(1:face_num, mesh.f(:,1), ones(face_num,1), face_num, vertex_num);
S2 = sparse(1:face_num, mesh.f(:,2), ones(face_num,1), face_num, vertex_num);
S3 = sparse(1:face_num, mesh.f(:,3), ones(face_num,1), face_num, vertex_num);

v = mesh.v;

v_struct = struct('f', v, 'dmesh_v', ones(vertex_num,3));



%grad_cell = cell(sample_num,1);
grad_total = zeros(1146*vertex_num*3,1);



for i = 1:size(direction,1),
    

   [grad] = angular_sampling_for_autodiff_grad(v_struct, S1, S2, S3, direction(i,:), lighting, sensor, lighting_normal, sensor_normal);
   
   grad_total = grad_total + grad.dmesh_v;
   
%    test = reshape(grad.dmesh_v, grad.dmesh_v_size);
%    test(find(grad.f), :,:)
   %[d, dy, dh] = checkgrad('f', v(:), 1e-5, S1, S2, S3, direction(i,:), lighting, sensor, lighting_normal, sensor_normal);
   %d
   %    
%    tmp = reshape(dh, grad.dmesh_v_size);
%    [test tmp]
%    tmp(find(grad.f),:,:)
   
   
   %find(dy)
   %[r] = checkgrad2(@f, v(:), {S1, S2, S3, direction(i,:), lighting, sensor, lighting_normal, sensor_normal});
   
   
end


grad_total = reshape(grad_total, grad.dmesh_v_size(1), length(grad_total)/grad.dmesh_v_size(1));

grad_total = 2*pi*grad_total/size(direction,1);
end