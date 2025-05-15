# ✅ Step 1: Install Required Libraries
# ✅ Step 2: Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json
import os

# ✅ Step 3: Define Dataset Paths
train_dir = "/content/drive/MyDrive/dataset/train"
val_dir = "/content/drive/MyDrive/dataset/test"

# ✅ Step 4: Data Preparation
img_size = (300, 300)  # EfficientNetB3 requires larger input
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ✅ Step 5: Build Model with EfficientNetB3
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False  # Freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ✅ Step 6: Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Step 7: Train the Model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# ✅ Step 8: Save the Model
model.save("efficientnetb3_tomato_model.h5")

# ✅ Step 9: Save Class Indices for Prediction
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

print("✅ EfficientNetB3 model and class indices saved successfully.")
