# ✅ Step 1: Install dependencies
# ✅ Step 2: Import libraries
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import numpy as np

# ✅ Step 3: Set up test dataset path and image parameters
test_dir = "/content/drive/MyDrive/dataset/test"
img_size = (300, 300)  # EfficientNetB3 uses 300x300
batch_size = 32

# ✅ Step 4: Load the trained EfficientNetB3 model and class indices
model = load_model("efficientnetb3_tomato_model.h5")

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Reverse the mapping for human-readable labels
class_labels = {v: k for k, v in class_indices.items()}

# ✅ Step 5: Create test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ✅ Step 6: Evaluate the model
loss, accuracy = model.evaluate(test_data)
print(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# ✅ Step 7: Make predictions and show results
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_data.classes

# ✅ Optional: Print a few predictions
for i in range(5):
    true_label = class_labels[true_classes[i]]
    predicted_label = class_labels[predicted_classes[i]]
    print(f"[Sample {i+1}] True: {true_label} | Predicted: {predicted_label}")
