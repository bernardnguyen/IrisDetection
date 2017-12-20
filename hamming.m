% Comparison via Hamming Distance

% The Hamming distance is the number of substitutions required to make the
% two inputs identical. To do this, we subtract one sample from the other and count the zeros. 
% We then divide by the size to get the percentage match.

function [matchPercent] = hamming(barcode1, barcode2)
total = size(barcode1,1) * size(barcode1,2);

totalMatch = abs(barcode1 - barcode2); 
matchPercent = 100*(length(find(totalMatch == 0))/total);
