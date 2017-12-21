% Formatted barcode for visualization

function [formatted] = formatBarcode(barcode)
[rows,cols] = size(barcode);

formatted = zeros(rows*3+2,cols+4);
for r = 1:rows
    formatted(r*3-2:r*3-1,:) = 200;
    formatted(r*3,3:cols+2) = 255*barcode(r,:);
    
    r = r+2;
end
formatted(rows*3+1:rows*3+2,:) = 200;
formatted(:,1:2) = 200;
formatted(:,cols+3:cols+4) = 200;