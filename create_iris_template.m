% Iris Detection: An Ideal Scenario

% In this project we explore the underlying concepts that bring about a
% naive implementation for iris detection. For this, we explore an ideal
% scenario where there is minimal blurring, obstructions (eyelids,lashes,etc),
% minimal glare, misalignment of pupil and iris, and higher quality. This
% case is presented in the file 'eye3.jpg'.  

function [barcode] = create_iris_template(file_name)
image = imread(file_name);
%image = imread('bernard_cropped.jpg');
%image = imread('eye3_cropped.jpg');
image = imresize(image, [300 300]);
[dummy_image, radii_pupil, radii_iris, x_pupil, y_pupil] = segmentation(image);
[normalized_image] = normalization(radii_pupil, radii_iris, dummy_image, x_pupil, y_pupil);
[barcode] = encoding(normalized_image);









