% Iris Detection: An Ideal Scenario

% In this project we explore the underlying concepts that bring about a
% naive implementation for iris detection. For this, we explore an ideal
% scenario where there is no blurring, obstructions (eyelids,lashes,etc),
% minimal glare, no misalignment of pupil and iris, and high quality. This
% case is presented in the file 'eye3.jpg'. 

clear; clc;
image = imread('eye2.jpg');

% Grayscaling is required for the rest of matlab functions
gray = rgb2gray(image);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Iris Segmentation %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Iris segmentation can be explained briefly as determining the boundaries
%   of the iris and pupil to obtain a donut shape that will be encoded.

% Smoothing with a gaussian filter is needed to determine 'true' edges
%   In this context, 'true' edges simply means 'discernable by humans'
%   which, for the case of iris detection, is the circles of iris and pupil
smoothed_im = imgaussfilt(gray,6.5);

% imfindcircles() uses circular Hough transform to determine circles
%   based off of radius thresholds(DEPENDENT ON IMAGE)

% Finding pupil boundaries
Rmin = 30;
Rmax = 50;
[centers_pupil, radii_pupil] = imfindcircles(smoothed_im,[Rmin Rmax],'ObjectPolarity','dark');

% Finding iris boundaries
Rmin = 50;
Rmax = 118;
[centers_iris, radii_iris] = imfindcircles(smoothed_im,[Rmin Rmax],'ObjectPolarity','dark');

% Display results in steps
figure(1);
subplot(2,3,1);
imshow(image);
title('original');
subplot(2,3,2);
imshow(gray);
title('grayscaled');
subplot(2,3,3);
imshow(smoothed_im);
title('Gaussian smoothed');
subplot(2,3,4);
imshow(smoothed_im);
viscircles(centers_pupil, radii_pupil,'Color','b');
title('finding pupil bounds');
subplot(2,3,5);
imshow(smoothed_im);
viscircles(centers_pupil(1,:), radii_iris(1,:),'Color','b');
title('finding iris bounds');
subplot(2,3,6);
imshow(image);
viscircles(centers_pupil, radii_pupil,'Color','b');
viscircles(centers_pupil(1,:), radii_iris(1,:),'Color','b');
title('segmentated image');

%%%%%%%%%%%%%%% Daugman Rubbersheet Modeling/Normalization %%%%%%%%%%%%%%%%

% Daugman Rubbersheet Modeling is a type of normalization that can be 
%   briefly described as taking the donut shape of the segmentated iris
%   and converting it into a rectangular form to be used for
%   normalization/encoding. Essentially, cut the donut from 0 degrees along
%   the horizontal until the pupil boundary is reached. Then you can
%   "unravel" the donut into a rectangular form!

% The first step in this process is to actually change pixel values of our
%   image based off the boundaries found in the segmentation step. This will
%   serve as markers in a later suppression step to properly store the iris
%   spectrum pixels into a rectangular format.

dummy_image = rgb2gray(imread('eye2.jpg'));

figure(2);
subplot(2,2,1);
imshow(dummy_image);
title('original');

center_pupil = ceil(centers_pupil);
x_pupil = center_pupil(2);
y_pupil = center_pupil(1);

theta = [0: pi/1000 : 2*pi];
xcoords_pupil = ceil(radii_pupil) * cos(theta) + x_pupil;
ycoords_pupil = ceil(radii_pupil) * sin(theta) + y_pupil;

for i = 1:size(xcoords_pupil,2)
    dummy_image(ceil(xcoords_pupil(i)),ceil(ycoords_pupil(i))) = 255;
end

center_iris = ceil(centers_pupil);
x_iris = center_iris(2);
y_iris = center_iris(1);

theta = [0: pi/2000 : 2*pi];
xcoords_iris = ceil(radii_iris(1)) * cos(theta) + x_iris;
ycoords_iris = ceil(radii_iris(1)) * sin(theta) + y_iris;

for i = 1:size(xcoords_iris,2)
    dummy_image(ceil(xcoords_iris(i)),ceil(ycoords_iris(i))) = 255;
end

subplot(2,2,2);
imshow(dummy_image);
title('pixels adjusted');

% Now that the dummy image's pixel's have been altered, we need to move
%   the iris spectrum pixels into a rectangular matrix. This will
%   eventually be our encoded template.

for i = 1:size(dummy_image,1)
    for j = 1:size(dummy_image,2)
        if (((i - x_pupil)^2 + (j - y_pupil)^2) <= (ceil(radii_pupil))^2)
            dummy_image(i,j) = 255;
        end
    end
end

subplot(2,2,3);
imshow(dummy_image);
title('inside pupil suppressed');

for i = 1:size(dummy_image,1)
    for j = 1:size(dummy_image,2)
        if (((i - x_iris)^2 + (j - y_iris)^2) >= (ceil(radii_iris(1)))^2)
            dummy_image(i,j) = 255;
        end
    end
end

subplot(2,2,4);
imshow(dummy_image);
title('outside iris suppressed');


% Now that the dummy image has been properly segmentated, it needs to be
%   stored into a matrix suitable for encoding. This step is important as
%   there needs to be a constant size of the template in order to properly
%   match. For this case we will be using a template of size ______.


% Insert normalization.

% apply Gabor filter
[G,GABOUT]=gaborfilter(dummy_image,0.05,0.025,0,0);

R=real(GABOUT); % REAL
I=imag(GABOUT); % IMAGINARY
M=abs(GABOUT); % MAGNITUDE
P=angle(GABOUT); % PHASE

% Set real and imaginary bits
R = uint8(R>=0); 
I = uint8(I>=0);

% Combine real and imaginary bits into one matrix with the structure:
% R11, R12, R13...
% I11, I12, I13...
% R21, R22, R23... etc.
[rows, cols] = size(R);
rbarcode = zeros(rows*2, cols);
cbarcode = zeros(rows, cols*2);
for r = 1:rows
    rbarcode(r*2-1,:) = R(r,:);
    rbarcode(r*2,:) = I(r,:);
    
    r = r+1;
end

for c = 1:cols
    cbarcode(:,c*2-1) = R(:,c);
    cbarcode(:,c*2) = I(:,c);
    
    c = c+1;
end

figure;
subplot(2,2,1); imshow(mat2gray(R)); title('Real part');
subplot(2,2,2); imshow(mat2gray(I)); title('Imaginary part');
subplot(2,2,3); imshow(mat2gray(rbarcode)); title('Row-stack "Barcode"');
subplot(2,2,4); imshow(mat2gray(cbarcode)); title('Column-stack "Barcode"');









