% Same as create_iris_template, but with extra outputs for visuals 

function [barcode, dummy_image, normalized_image] = create_iris_template_visual(file_name)
image = imread(file_name);
image = imresize(image, [300 300]);
[dummy_image, radii_pupil, radii_iris, x_pupil, y_pupil] = segmentation(image);
[normalized_image] = normalization(radii_pupil, radii_iris, dummy_image, x_pupil, y_pupil);
[barcode] = encoding(normalized_image);
