# Facial-Emotion-Recognition

This repository contains a Convolutional Neural Network (CNN) model for recognizing facial emotions. The model is trained on the FER-2013 dataset from Kaggle.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)

## Overview

Facial Emotion Recognition is a task of recognizing the emotions expressed on human faces. This project uses a CNN model to classify images of faces into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Dataset

The FER-2013 dataset consists of 35,887 grayscale, 48x48 sized face images with seven emotion labels: 
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

You can download the dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Model Architecture

The CNN model architecture used for this project is as follows:
- **Input Layer**: 48x48 grayscale images
- **Convolutional Layers**: Multiple layers with ReLU activation and MaxPooling
- **Fully Connected Layers**: Dense layers with Dropout for regularization
- **Output Layer**: Softmax layer for classification

## Installation

To run this project, you'll need Python and the following libraries:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install the necessary libraries using pip:

```sh
pip install tensorflow keras numpy pandas matplotlib
