# Potato-Disease-Detection

## Introduction
This project aims to detect whether potato leaves are healthy or have a disease using Convolutional Neural Networks (CNN). The project includes image preprocessing, 
model training, API creation using FastAPI, and deployment on Google Cloud Platform (GCP).

## DataSet
The dataset used for this project contains images of potato leaves, labeled as healthy or diseased. The dataset is divided into training, 
validation, and testing sets to ensure accurate model evaluation.

## Preprocessing
Before feeding the images to the model, preprocessing steps were applied, including:
* Resizing images to a standard size.
* Normalizing pixel values to [0, 1] range.
* Augmenting the dataset for better generalization.

## Model
The CNN model was designed to classify potato leaves as healthy or diseased. It consists of multiple convolutional and pooling layers, 
followed by fully connected layers. The model was trained using the training dataset and validated on the validation set to optimize performance.

## API using FastAPI
FastAPI was used to create a web API to expose the trained model's predictions. The API receives images as input and returns the predicted class 
(healthy or diseased) along with a confidence score.

## Deployment on Google Cloud Platform (GCP)
The trained model and the FastAPI app were deployed on Google Cloud Platform (GCP) to make the predictions accessible
over the internet. The API endpoints can be accessed through a public URL.
