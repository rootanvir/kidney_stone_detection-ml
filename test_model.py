import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ================================
# Step 1: Load the model
# ================================
model = load_model("trained_model.h5")  # your saved model

# ================================
# Step 2: Load & preprocess image
# ================================
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # resize
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # add batch dim
    img_array = img_array / 255.0  # normalize

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction, axis=1)[0]

    return class_index, prediction

# ================================
# Step 3: Class labels (match training)
# ================================
class_labels = {0: "Non-Stone", 1: "Stone"}  # adjust if reversed

# ================================
# Step 4: Test
# ================================
test_image = "images/stone.jpg"  # replace with your image path
idx, prob = predict_image(test_image)

print(f"Predicted: {class_labels[idx]} ({prob[0][idx]*100:.2f}%)")
