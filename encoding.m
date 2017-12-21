%%%%%%%%%%%%%%%%%%%%%%%% Encoding/Feature Extraction %%%%%%%%%%%%%%%%%%%%%%

function [barcode] = encoding(normalized_image)
% apply Gabor filter
S = 0.15; % Variance
F = 0.001; % Polar frequency
W = 0; % 
P = 0; % Phase
[G,GABOUT]=gaborfilter(normalized_image,S,F,W,P);

R=real(GABOUT); % REAL
I=imag(GABOUT); % IMAGINARY
M=abs(GABOUT); % MAGNITUDE
P=angle(GABOUT); % PHASE

% Set real and imaginary bits
R = uint8(R>=0)*1; 
I = uint8(I>=0)*1;

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