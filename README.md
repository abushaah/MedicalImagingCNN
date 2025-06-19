# MedicalImagingCNN

Haifaa, May 2025

## Abstract

**About**

Using convolutional neural networks (ML) for classifying medical images.

**What does the software do?**

Analyze, detect, and diagnose various tumours depending on the dataset trained

**How?**

Trained on existing positive and negative medical images

**Purpose**

- For doctors:
    - Prioritize scans on a daily basis for viewing by doctor
    - Prioritize scans based on medical history of patient and result
- For patients:
    - Faster results

## Model

### U-net
- A deep learning architecture for semantic segmentation
    - Recall: Semantic segmentation is where each pixel in an image is classified and labeled with a category or class. Instance segmentation specifies how many instances of the class is in the segmented image
- Created for biomedical imaging
    - Only require semantic segmentation since there is only one instance of the organ in the medical image
    - 2 classes are identified, one organ and another for the background
    - Value closer to 1 means it is closer to being identified as the class
- Divided into 2 parts: Encoder and Dencoder
    - Encoder: A series of convolutional layers (3x3), RelU, max pool 2x2
    - Dencoder: Build output image from encoder to classify each image

### Training

### Testing

## Software tools

### Monai
- A Medical Open Network for Artificial Intelligence using Dicom/Nifti files
- Each epoch will crop a random window in the image for training

### Python
- Pytorch

## Data Sources
**Note: all data sources have a reliability score of 8.5+ on kaggle**

- [Brain tumour MRI](https://www.kaggle.com/datasets/andrewmvd/brain-tumor-segmentation-in-mri-brats-2015)
- [Breast cancer cell](https://www.kaggle.com/datasets/andrewmvd/breast-cancer-cell-segmentation)
- [Hippocampus MRI](https://www.kaggle.com/datasets/andrewmvd/hippocampus-segmentation-in-mri-images)
- [Lung vessel](https://www.kaggle.com/datasets/andrewmvd/lung-vessel-segmentation)
- [Liver tumour](https://www.kaggle.com/datasets/andrewmvd/liver-tumor-segmentation)

### Data preprocessing
1. NIFTI files
- Crop Nifti files to same dimensions
- For each patient, the DCM files in the nifti file must be equal
- Compress files

# Validation
1. Use an AI model evaluation (with a focus on healthcare): [epic-open-source/seismometer](https://github.com/epic-open-source/seismometer)
2. Consult in industry experts for their opinions

## To do
1. Base project which can work on any dataset
2. Add multiple data sets
3. Add reasoning process to the result
4. Create a UI for uploading datasets and obtain results

### Research papers, tutorials, and resources
- [Monai](https://monai.io/index.html)
- [Kaggle](https://www.kaggle.com/)
