
# Moiré Pattern Detection Optimization

This project builds upon the methodologies outlined in "Moiré Pattern Detection Using Wavelet Decomposition and Convolutional Neural Network," introducing key optimizations to address challenges encountered with large datasets and complex image backgrounds.

## Key Challenges & Solutions


### Model Enhancement

Revisited the previously discarded HH band information from the Wavelet decomposition. The inclusion of the HH band, after normalization with the LL band, significantly improved the model's performance. Consequently, the revised model now accepts four inputs (LL, LH, HL, and HH bands) instead of the initial three.


### Efficient Data Handling with Generators

**Problem**:  The extensive memory requirement to process large datasets, with the initial approach rendered traditional data loading methods impractical.

**Solution**: Overhauled the data handling mechanism by implementing generators and batch learning. This approach enabled dynamic data loading and processing, markedly increasing computational efficiency.


### Accurate Moiré Pattern Detection

**Problem 1**: The presence of strong textured backgrounds in images, which often mimic Moiré patterns, resulted in a high number of false positives—normal images misclassified as containing Moiré patterns.

**Solution 1**: Introduced a dynamic threshold adjustment to fine-tune detection sensitivity, effectively reducing false positives while ensuring the accurate identification of true positives.

**Problem 2**: Loss of crucial Moiré pattern details during the downscaling of images for training adversely affected the algorithm's detection capability.

**Solution 2**: A multi-faceted approach was adopted to preserve pattern details:
	-	**Cropping Strategies**:
	-	 - **Image Crop**: Center cropping to provide a standard image processing technique.
  - **CropBrightest**: Brightness-based cropping to focus on the brightest areas of images, where pattern details are most pronounced, thereby enhancing pattern preservation.
- **Optimized Image Processing**: Investigated various interpolation methods and experimented with larger image sizes, constrained by computational resources, to minimize pattern degradation.



## Setup Instructions

### Prerequisites

- Python 3

### Install Dependencies


```bash
pip install tensorflow keras Pillow scikit-learn scikit-image
```

## Data Preparation

### Setting Universal Constants

Define the image width and height in your script. Larger dimensions can improve detection accuracy but require more computational resources.

### Creating the Dataset

Generate your training dataset using the provided script:

```bash
python createTrainingData.py <positiveImages> <negativeImages> <mode>
```

- `positiveImages`: Path to the directory containing images with Moiré patterns.
- `negativeImages`: Path to the directory with normal images.
- `mode`: Use `0` for training mode or `1` for test mode.

### Image Cropping

To optimize processing time for high-resolution images, consider cropping them to retain essential details:
- **Image Crop**: Crops from the center, suitable for general purposes.
- **CropBrightest**: Targets the brightest area in the image, enhancing Moiré pattern detection.

Execute the `createTrainingData.py` script after cropping to augment and decompose the dataset, preparing it for training.

## Training the Model

Before training, ensure all constants and required imports are correctly set up.

To start training, use the following command, specifying the paths to your processed image data:

```bash
python trainModel.py <positiveTrainImagePath> <negativeTrainImagePath>
```

## Testing the Model

For testing, use the transformed image folders similar to those used in the training phase. The threshold value is optional and can be adjusted to fine-tune sensitivity and reduce false positives, but it's not required to run the test:

```bash
python testModel.py <positiveTestImagePath> <negativeTestImagePath> [<threshold>]


- The saved CNN model will be loaded automatically.
- Adjust the `load_model` function in the script if you need to use a different model.
