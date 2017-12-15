%%%%%%%%%%%%%%%%%%%%%%%% Encoding/Feature Extraction %%%%%%%%%%%%%%%%%%%%%%

function [barcode] = encoding(normalized_image)
% apply Gabor filter
S = 0.4; % Variance
F = 0.025; % Polar frequency
W = 0; % 
P = 0; % Phase
[G,GABOUT]=gaborfilter(normalized_image,S,F,W,P);

R=real(GABOUT); % REAL
I=imag(GABOUT); % IMAGINARY
M=abs(GABOUT); % MAGNITUDE
P=angle(GABOUT); % PHASE

% Set real and imaginary bits
R = uint8(R>=0)*255; 
I = uint8(I>=0)*255;

% Combine real and imaginary bits into one matrix with the structure:
% Row-stack: 
% R11, R12, R13...
% I11, I12, I13...
% R21, R22, R23...  and similarly for column stack.
[rows, cols] = size(R);
barcode = zeros(rows,cols*2);

for c = 1:cols
    barcode(:,c*2-1) = R(:,c);
    barcode(:,c*2) = I(:,c);
    
    c = c+1;
end

[rows,cols] = size(barcode);

formatted = zeros(rows*3+2,cols+4);
for r = 1:rows
    formatted(r*3-2:r*3-1,:) = 200;
    formatted(r*3,3:cols+2) = barcode(r,:);
    
    r = r+2;
end
formatted(rows*3+1:rows*3+2,:) = 200;
formatted(:,1:2) = 200;
formatted(:,cols+3:cols+4) = 200;

figure; 
subplot(211); imshow(mat2gray(barcode));
subplot(212); imshow(mat2gray(formatted));