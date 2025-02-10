Deepfake Detection using AI

📌 Project Overview

This project aims to train a model for detecting deepfake videos using artificial intelligence. It leverages deep learning techniques such as Convolutional Neural Networks (CNNs) and Vision Transformers to analyze video frames and detect manipulations.

🎯 Objectives

Detect manipulated facial features and inconsistencies.

Differentiate between real and AI-generated videos.

Implement adversarial training for robustness.

📂 Dataset

We use two well-known datasets for training and evaluation:

FaceForensics++

Deepfake Detection Challenge (Kaggle)

🛠️ Techniques & Tools

Machine Learning Models

CNNs & Vision Transformers (EfficientNet, Swin Transformer)

Spatio-Temporal Analysis (LSTMs, 3D-CNNs)

Deepfake Detection Pipelines (XceptionNet, MTCNN for face tracking)

Libraries Used

OpenCV

TensorFlow/Keras

DeepFaceLab

MediaPipe

NumPy, Pandas, Scikit-learn

🚀 Implementation Steps

Data Preprocessing

Extract frames from videos.

Normalize and resize images.

Perform face tracking using MTCNN.

Feature Extraction

Use InceptionV3 or EfficientNet for feature extraction.

Model Training

Train CNN-based classifiers.

Experiment with LSTMs & 3D-CNNs for temporal analysis.

Evaluation & Testing

Test models on unseen deepfake videos.

Use accuracy, precision, recall, and F1-score for evaluation.

⚡ How to Run

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook in Google Colab:

jupyter notebook deepFake.ipynb

Upload your dataset and start training.

📝 Notes

Ensure that you have Google Colab Pro if using a large dataset for faster training.

If mtcnn is missing, install it manually:

pip install mtcnn

If numpy throws np.int deprecation warnings, replace np.int with int in your code.

📧 Contact

For any queries, reach out via email or GitHub Issues.