Pakistani Sign Language Recongnition using LSTM

**Project Overview**


This project focuses on gesture recognition using deep learning (LSTM). The model is trained on a dataset consisting of different gesture classes (no, good, late) using keypoints extracted from videos. The LSTM model learns temporal patterns from sequences of frames and classifies gestures accurately.

**Dataset Structure**

The dataset is structured as follows:

MP_Data/
    no/
        1.npy
        2.npy
        ...
        30.npy
    good/
        1.npy
        2.npy
        ...
        30.npy
    late/
        1.npy
        2.npy
        ...
        30.npy

There are 3 classes: no, good, and late.

Each class contains exactly 30 videos.

From each video, 30 frames were extracted and preprocessed into .npy format.

Each .npy file contains keypoints extracted using Mediapipe.



**Keypoint Extraction**

MediaPipe Holistic extracts the following keypoints:

Face Mesh Landmarks (468 points)

Pose Landmarks (33 points)

Left Hand Landmarks (21 points)

Right Hand Landmarks (21 points)

These keypoints are used as input to the LSTM model for action recognition.



**Model Architecture**

The model consists of three LSTM layers to capture temporal dependencies, followed by dense layers for classification:

LSTM(64): Captures short-term patterns from the sequences.

LSTM(128): Enhances feature extraction by learning deeper temporal dependencies.

LSTM(64): Further refines the learned patterns.

Dense(64, ReLU): Fully connected layer for feature processing.

Dense(32, ReLU): Additional dense layer for feature transformation.

Dense(Softmax): Output layer for classification.



**Results & Evaluation**

After training the model with 5-fold cross-validation, the results are as follows:

Mean Accuracy: 90%

Final Validation Accuracy: 94.44%



**Confusion Matrix:**

array([[[11,  0],
        [ 1,  6]],

       [[13,  1],
        [ 0,  4]],

       [[11,  0],
        [ 0,  7]]], dtype=int64)
