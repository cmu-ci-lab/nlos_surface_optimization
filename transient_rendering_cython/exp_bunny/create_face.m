function [face] = create_face(mask)
total_num = sum(mask(:));
index = nan(size(mask));
index(mask(:))=1:total_num;

face = [];
for i = 2:size(mask,1)
    for j = 2:size(mask,2)
        if mask(i,j) == 0
            if mask(i-1,j-1)==1 && mask(i-1,j)==1 && mask(i,j-1)==1
                face = [face; [index(i-1,j-1) index(i-1,j) index(i,j-1)]];
            end
            
        else
            if mask(i-1, j -1) == 0
                if mask(i-1, j) == 1 && mask(i,j-1)==1
                    face = [face; [index(i-1,j) index(i,j) index(i,j-1)]];
                    
                end
            else
                if mask(i-1, j) == 1
                    face = [face; [index(i-1,j-1) index(i-1,j) index(i,j)]];
                end
                if mask(i, j-1) == 1
                    face = [face; [index(i-1,j-1) index(i,j) index(i,j-1)]];
                end
            end
        end
        
        
    end
end


end