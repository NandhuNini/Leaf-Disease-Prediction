# Leaf-Disease-Prediction
Tomato Leaf Disease Detection using EfficientNetB3 and Deep Learning
 Objective:
To build an AI-powered image classification model that accurately detects and identifies tomato leaf diseases from images using EfficientNetB3. The model helps farmers, agriculturists, and researchers to diagnose diseases early, enabling timely intervention and improved crop yield.

Technologies & Libraries Used:
Python
TensorFlow / Keras
EfficientNetB3
ImageDataGenerator (for augmentation)
NumPy & PIL
Gradio (for UI deployment)

 1. Model Training (EfficientNetB3):
Used EfficientNetB3 pre-trained on ImageNet for feature extraction.
Replaced the top classifier with custom layers for multi-class classification.
Applied image augmentation (rotation, zoom, flip, etc.) using ImageDataGenerator.
Trained for 10+ epochs using categorical cross-entropy and Adam optimizer.
Saved the model as efficientnetb3_tomato_model.h5.
Saved the class index mapping to class_indices.json.

✅ 2. Model Testing:
Loaded the trained model and class mappings.
Preprocessed test dataset using ImageDataGenerator.
Evaluated model performance on unseen data.
Printed accuracy and sample predictions to validate output quality.

✅ 3. Real-Time Prediction with Gradio:
Created a Gradio UI where users can upload a tomato leaf image.
Preprocesses the uploaded image to 300x300 (as required by EfficientNetB3).
Predicts disease class with a confidence score.
Displays results in an interactive and user-friendly web interface.
Supports share=True for public accessibility via URL (ideal for demos and testing).

🧪 Model Performance (Example):
Test Accuracy: ~95% (Varies depending on dataset size and quality)
Fast and accurate predictions using EfficientNetB3
Good generalization due to data augmentation and pre-trained weights.
