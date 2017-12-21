%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Now that the dummy image has been properly segmentated, it needs to be
%   stored into a matrix suitable for encoding. This step is important as
%   there needs to be a constant size of the template in order to properly
%   match. For this case we will be using a template of size 240x20.

function [normalized_image] = normalization(radii_pupil, radii_iris, dummy_image, x_pupil, y_pupil)

% Crop image to center the iris
radius = ceil(radii_iris(1));
xTop = x_pupil - radius;
width = xTop + 2 * radius;
yLeft = y_pupil - radius;
height = yLeft + 2 * radius;
dummy_image = imcrop(dummy_image, [yLeft, xTop, width, height]);

% Normalize
radial_res = 20; % M
angular_res = 75; % N 

rMax = 1;
rMin = radii_pupil(1) / radii_iris(1);

normalized_image = double(dummy_image)/255.0;
normalized_image = ImToPolar(normalized_image, rMin, rMax, radial_res, angular_res);

%figure;
%subplot(2,1,1)
%imshow(dummy_image); title('Segmented iris');
%subplot(2,1,2)
%imshow(normalized_image); title('Normalized iris');