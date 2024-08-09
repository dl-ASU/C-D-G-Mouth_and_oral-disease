Dataset Analysis and Transformation
Overview
This repository includes a script for analyzing and transforming a dataset of images. The CustomDataset class is used to load and preprocess images, and the analyze_dataset method provides descriptive statistics on the dataset, including pixel intensity statistics and label distribution.

Dataset Description
The dataset contains images organized into categories and regions:

Categories
high
low
normal
Regions
buccal_mucosa_left
buccal_mucosa_right
dorsum_of_tongue
floor_of_mouth
gingive
lateral_broder_of_tongue_left
lateral_broder_of_tongue_right
lower_labial_mucosa
palate
upper_labial_mucosa
ventral_of_tongue
Transformations Applied
The images are transformed using the following steps:

Resize: Images are resized to 128x128 pixels.
Normalization: The images are converted to floating-point values in the range [0, 1].
Histogram Equalization: Each color channel undergoes histogram equalization to enhance contrast.
Convert to 8-bit: The processed images are scaled to 8-bit unsigned integers.
Convert to Tensor: The final images are converted to PyTorch tensors.
Descriptive Statistics
The analyze_dataset method calculates the following statistics:

Pixel Intensity Statistics
Mean Pixel Value per Channel:

Red: 0.5027
Green: 0.5017
Blue: 0.5018
Standard Deviation of Pixel Value per Channel:

Red: 0.2904
Green: 0.2886
Blue: 0.2886
Variance of Pixel Value per Channel:

Red: 1.0469e-06
Green: 1.4237e-07
Blue: 1.3100e-07
Label Distribution
High: 2052 samples (24.31%)
Low: 2051 samples (24.30%)
Normal: 4337 samples (51.39%)
Pearson's Correlation Coefficient
Pearson's R between Mean and Standard Deviation of Pixel Values: 0.78
This indicates a moderate positive correlation between the mean and standard deviation of pixel values across the dataset.