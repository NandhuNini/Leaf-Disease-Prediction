# ✅ Step 2: Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import json
import gradio as gr

# ✅ Step 3: Load EfficientNetB3 model and class labels
model = load_model("efficientnetb3_tomato_model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the mapping
class_labels = {v: k for k, v in class_indices.items()}

# ✅ Step 4: Prediction function
def predict(image):
    image = image.convert("RGB")
    image = image.resize((300, 300))  # EfficientNetB3 expects 300x300
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_class = class_labels[class_index]

    return f"🔍 Predicted: {predicted_class}\n📊 Confidence: {confidence:.2f}%"

# ✅ Step 5: Launch Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Tomato Leaf Image"),
    outputs="text",
    title="🍅 Tomato Leaf Disease Detector (EfficientNetB3)",
    description="Upload a tomato leaf image to identify the disease using AI",
    theme="default"
)

interface.launch(share=True)
