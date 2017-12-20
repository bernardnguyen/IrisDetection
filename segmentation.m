%%%%%%%%%%%%%%%%%%%%%%%%%%%% Iris Segmentation %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Iris segmentation can be explained briefly as determining the boundaries
%   of the iris and pupil to obtain a donut shape that will be encoded.

% Smoothing with a gaussian filter is needed to determine 'true' edges
%   In this context, 'true' edges simply means 'discernable by humans'
%   which, for the case of iris detection, is the circles of iris and pupil

function [dummy_image, radii_pupil, radii_iris, x_pupil, y_pupil] = segmentation(image)

gray = rgb2gray(image);
smoothed_im = imgaussfilt(gray,6.5);

% imfindcircles() uses circular Hough transform to determine circles
%   based off of radius thresholds(DEPENDENT ON IMAGE)

% Finding pupil boundaries 
Rmin = 40;
Rmax = 70;
[centers_pupil, radii_pupil] = imfindcircles(smoothed_im,[Rmin Rmax],'ObjectPolarity','dark');

% Finding iris boundaries 
Rmin = 100;
Rmax = 250;
[centers_iris, radii_iris] = imfindcircles(smoothed_im,[Rmin Rmax],'ObjectPolarity','dark');

% Display results in steps
%figure;
%subplot(2,3,1);
%imshow(image);
%title('original');
%subplot(2,3,2);
%imshow(gray);
%title('grayscaled');
%subplot(2,3,3);
%imshow(smoothed_im);
%title('Gaussian smoothed');
%subplot(2,3,4);
%imshow(smoothed_im);
%viscircles(centers_pupil(1,:), radii_pupil(1,:),'Color','b');
%title('finding pupil bounds');
%subplot(2,3,5);
%imshow(smoothed_im);
%viscircles(centers_pupil(1,:), radii_iris(1,:),'Color','b');
%title('finding iris bounds');
%subplot(2,3,6);
%imshow(image);
%viscircles(centers_pupil(1,:), radii_pupil(1,:),'Color','b');
%viscircles(centers_pupil(1,:), radii_iris(1,:),'Color','b');
%title('segmentated image');

dummy_image = gray;

%figure;
%subplot(2,2,1);
%imshow(dummy_image);
%title('original');

center_pupil = ceil(centers_pupil(1,:));
x_pupil = center_pupil(2);
y_pupil = center_pupil(1);

theta = [0: pi/1000 : 2*pi];
xcoords_pupil = ceil(radii_pupil(1,:)) * cos(theta) + x_pupil;
ycoords_pupil = ceil(radii_pupil(1,:)) * sin(theta) + y_pupil;

for i = 1:size(xcoords_pupil,2)
    dummy_image(ceil(xcoords_pupil(i)),ceil(ycoords_pupil(i))) = 255;
end

center_iris = ceil(centers_pupil(1,:));
x_iris = center_iris(2);
y_iris = center_iris(1);

theta = [0: pi/2000 : 2*pi];
xcoords_iris = ceil(radii_iris(1)) * cos(theta) + x_iris;
ycoords_iris = ceil(radii_iris(1)) * sin(theta) + y_iris;

for i = 1:size(xcoords_iris,2)
    if ((xcoords_iris(i) > 0) && (ycoords_iris(i) > 0) && (xcoords_iris(i) <= size(image,1)) && (ycoords_iris(i) <= size(image,1))) 
        dummy_image(ceil(xcoords_iris(i)),ceil(ycoords_iris(i))) = 255;
    end
end

%subplot(2,2,2);
%imshow(dummy_image);
%title('pixels adjusted');

% Now that the dummy image's pixel's have been altered, we need to move
%   the iris spectrum pixels into a rectangular matrix. This will
%   eventually be our encoded template.

for i = 1:size(dummy_image,1)
    for j = 1:size(dummy_image,2)
        if (((i - x_pupil)^2 + (j - y_pupil)^2) <= (ceil(radii_pupil(1,:)))^2)
            dummy_image(i,j) = 255;
        end
    end
end

%subplot(2,2,3);
%imshow(dummy_image);
%title('inside pupil suppressed');

for i = 1:size(dummy_image,1)
    for j = 1:size(dummy_image,2)
        if (((i - x_iris)^2 + (j - y_iris)^2) >= (ceil(radii_iris(1)))^2)
            dummy_image(i,j) = 255;
        end
    end
end

%subplot(2,2,4);
%imshow(dummy_image);
%title('outside iris suppressed');
