% Matching test demo
clear;clc;
barcode_eye1 = create_iris_template('eye3_cropped.jpg');
barcode_eye2 = create_iris_template('bernard_cropped.jpg');
barcode_eye3 = create_iris_template('bernard_cropped2.jpg');
barcode_eye4 = create_iris_template('bernard_cropped3.jpg');
barcode_eye5 = create_iris_template('high_guy1_cropped.jpg');
barcode_eye6 = create_iris_template('high_guy2_cropped.jpg');
barcode_eye7 = create_iris_template('other_guy1_cropped.jpg');
barcode_eye8 = create_iris_template('other_guy2_cropped.jpg');

total = size(barcode_eye1,1) * size(barcode_eye1,2);

percentage_match1 = hamming(barcode_eye1,barcode_eye2); % original to bernard1
percentage_match2 = hamming(barcode_eye2,barcode_eye3); % bernard1 to bernard2
percentage_match3 = hamming(barcode_eye3,barcode_eye4); % bernard2 to bernard3
percentage_match4 = hamming(barcode_eye3,barcode_eye5); % bernard2 to highguy1
percentage_match5 = hamming(barcode_eye3,barcode_eye6); % bernard2 to highguy2
percentage_match6 = hamming(barcode_eye5,barcode_eye6); % highguy1 to highguy2
percentage_match7 = hamming(barcode_eye3,barcode_eye7); % bernard2 to otherguy1
percentage_match8 = hamming(barcode_eye3,barcode_eye8); % bernard2 to otherguy2
percentage_match9 = hamming(barcode_eye7,barcode_eye8); % otherguy1 to otherguy2

disp([percentage_match1 percentage_match2 percentage_match3 percentage_match4 ...
    percentage_match5 percentage_match6 percentage_match7 percentage_match8 ...
    percentage_match9]);

%{
figure;
subplot(3,3,1);
imshow(mat2gray(barcode_eye1));
title('eye1 barcode');
subplot(3,3,2);
imshow(mat2gray(barcode_eye2));
title('eye2 barcode');
subplot(3,3,3);
imshow(mat2gray(matching_1));
title('matching difference');

subplot(3,3,4);
imshow(mat2gray(barcode_eye1));
title('eye1 barcode');
subplot(3,3,5);
imshow(mat2gray(barcode_eye1));
title('eye1 barcode');
subplot(3,3,6);
imshow(mat2gray(matching_2));
title('matching difference');

subplot(3,3,7);
imshow(mat2gray(barcode_eye2));
title('eye2 barcode');
subplot(3,3,8);
imshow(mat2gray(barcode_eye2));
title('eye2 barcode');
subplot(3,3,9);
imshow(mat2gray(matching_3));
title('matching difference');
%}