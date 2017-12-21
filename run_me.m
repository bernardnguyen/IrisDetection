% Run for final demo

% Create template for 4 test images:
clear;clc;
[b1, s1, n1] = create_iris_template_visual('bernard_cropped.jpg');
[b2, s2, n2] = create_iris_template_visual('bernard_cropped2.jpg');
[b3, s3, n3] = create_iris_template_visual('bernard_cropped3.jpg');
[b4, s4, n4] = create_iris_template_visual('high_guy1_cropped.jpg');

% Dimensions of the barcode:
total = size(b1,1) * size(b1,2);

% Matching
percentage_match1 = hamming(b1,b1); % bernard1 to bernard1
percentage_match2 = hamming(b1,b2); % bernard1 to bernard2
percentage_match3 = hamming(b1,b3); % bernard1 to bernard3
percentage_match4 = hamming(b1,b4); % bernard1 to highguy

correlation1 = corr2(b1,b1); % bernard1 to bernard1
correlation2 = corr2(b1,b2); % bernard1 to bernard2
correlation3 = corr2(b1,b3); % bernard1 to bernard3
correlation4 = corr2(b1,b4); % bernard1 to highguy


% Graphic output
bernard1 = imread('bernard_cropped.jpg');
bernard1 = imresize(bernard1, [300 300]);
bernard2 = imread('bernard_cropped2.jpg');
bernard2 = imresize(bernard2, [300 300]);
bernard3 = imread('bernard_cropped3.jpg');
bernard3 = imresize(bernard3, [300 300]);
other = imread('other_guy1_cropped.jpg');
other = imresize(other, [300 300]);

% Figure organization
figure;

% Comparison image - Bernard1
subplot(4,6,1); imshow(bernard1); title('Comparison Image');

% Original images
subplot(4,6,2); imshow(bernard1); title('Bernard1');
subplot(4,6,8); imshow(bernard2); title('Bernard2');
subplot(4,6,14); imshow(bernard3); title('Bernard3');
subplot(4,6,20); imshow(other); title('Other Iris');

% Segmented
subplot(4,6,3); imshow(s1); title('Segmented Iris');
subplot(4,6,9); imshow(s2); 
subplot(4,6,15); imshow(s3); 
subplot(4,6,21); imshow(s4); 

% Normalized
subplot(4,6,4); imshow(n1); title('Normalized Iris');
subplot(4,6,10); imshow(n2); 
subplot(4,6,16); imshow(n3); 
subplot(4,6,22); imshow(n4); 

% Barcode
subplot(4,6,5); imshow(mat2gray(formatBarcode(b1))); title('Iris Codes');
subplot(4,6,11); imshow(mat2gray(formatBarcode(b2))); 
subplot(4,6,17); imshow(mat2gray(formatBarcode(b3)));
subplot(4,6,23); imshow(mat2gray(formatBarcode(b4))); 

% Text output
ax1 = subplot(4, 6, 6);
text(0,1,{'Match % by Hamming Distance: ', percentage_match1}); 
text(0,0.5,{'Correlation: ', correlation1});
set (ax1, 'visible', 'off')

ax2 = subplot(4, 6, 12);
text(0,1,{'Match % by Hamming Distance: ', percentage_match2}); 
text(0,0.5,{'Correlation: ', correlation2});
set (ax2, 'visible', 'off')

ax3 = subplot(4, 6, 18);
text(0,1,{'Match % by Hamming Distance: ', percentage_match3}); 
text(0,0.5,{'Correlation: ', correlation3});
set (ax3, 'visible', 'off')

ax4 = subplot(4, 6, 24);
text(0,1,{'Match % by Hamming Distance: ', percentage_match4}); 
text(0,0.5,{'Correlation: ', correlation4});
set (ax4, 'visible', 'off')




