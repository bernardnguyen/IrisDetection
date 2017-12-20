% Normalized correlation

% Takes 2 barcodes as inputs, then calculates the normalized correlation
% between them for an output.

function [matchPercent] = correlation(barcode1, barcode2)

% Size
[rows, cols] = size(barcode1);

% Mean and Std. Dev.
m1 = mean(barcode1);
m2 = mean(barcode2);

s1 = std2(barcode1);
s2 = std2(barcode2);

% Normalized correlation
barcode1 = barcode1-m1;
barcode2 = barcode2-m2;

product = barcode1.*barcode2;
sum = sum(sum(product));

matchPercent = sum/(rows*cols*s1*s2);