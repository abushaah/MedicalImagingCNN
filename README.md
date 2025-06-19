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
- 3DSlicer

## Data Sources
**Note: all data sources have a reliability score of 8.5+ on kaggle**

- Gastrointestinal disease: [Kvasir Dataset](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)
- (Liver, brain, lung, pancreas) tumour, hippocampus, prostate, cardiac, colon cancer, hepatic vessels, spleen: [Medical Segmentation Decathlon](http://medicaldecathlon.com/)
- Pneumonia chest x ray: [Medical Image Analysis with CNN](https://www.kaggle.com/code/ghitabenjrinija/medical-image-analysis-with-cnn)
- Mammogram:
    - [RSNA Screening Mammography Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data?select=sample_submission.csv)
    - [King Abdulaziz University Mammogram Dataset](https://www.kaggle.com/datasets/asmaasaad/king-abdulaziz-university-mammogram-dataset)

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
- [Deep convolutional neural network based medical image classification for disease diagnosis](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0276-2#Sec6)
- [Medical Image Analysis with CNN](https://www.kaggle.com/code/ghitabenjrinija/medical-image-analysis-with-cnn)
- [PyTorch and Monai for AI Healthcare Imaging - Python Machine Learning Course](https://www.youtube.com/watch?v=M3ZWfamWrBM)

### Similar models
- BioMedCLIP
- CheXNet
