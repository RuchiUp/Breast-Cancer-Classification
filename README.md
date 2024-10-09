# COVID-19 Chest X-ray Classification
This project aims to classify chest X-ray images into three categories: COVID-19, pneumonia, and normal. The approach utilizes image processing, machine learning, and evaluation techniques to build and assess a classification model. This project implements a machine learning pipeline to classify chest X-ray images. The classification is performed using different models, including Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) with Principal Component Analysis (PCA) for feature extraction

## Project Description

The project aims to classify chest X-ray images to aid in the diagnosis of COVID-19. It utilizes image preprocessing techniques, 
various machine learning algorithms, and performance evaluation metrics to ensure accurate classification.

## Installation

To run this project, ensure you have Python installed. You can install the required libraries using pip.
```bash
pip install numpy pandas opencv-python matplotlib seaborn scikit-learn tensorflow


The dataset used in this project consists of chest X-ray images categorized into three classes:
COVID-19: Images of patients confirmed with COVID-19.
Normal: Images of healthy lungs.
Pneumonia: Images of patients with pneumonia.

The dataset is structured as follows:
dataset/
│
├── COVID_19/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── NORMAL/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── PNEUMONIA/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
'''

## Preprocessing: Images are resized to 224x224 pixels, converted to grayscale, and normalized to prepare them for analysis.
DataFrames are created to organize the images and their corresponding labels.

## Visualization: Sample images are displayed to verify the correctness of data loading and preprocessing.

## Data Splitting: The dataset is divided into training and testing sets in an 80-20 ratio to facilitate model training and evaluation.

## Model Development:

A Decision Tree Classifier is initially trained and evaluated for accuracy using metrics like confusion matrices and classification reports.
A K-Nearest Neighbors (KNN) classifier is implemented to compare performance.
Principal Component Analysis (PCA) is employed for dimensionality reduction, allowing for better model efficiency and performance.
A Support Vector Machine (SVM) model is trained with hyperparameter tuning using GridSearchCV for optimal results.
Evaluation and Visualization: The models are assessed using various metrics, and PCA components,
along with explained variance ratios, are visualized to understand the data distribution and model performance.

