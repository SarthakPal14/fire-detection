import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "fire_model.h5")
model = tf.keras.models.load_model(model_path)

# Allow passing an image path as a command-line argument, or default to a local test image
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    img_path = sys.argv[1]
else:
    img_path = os.path.join(BASE_DIR, "test99.jpg")


img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)


prediction = float(model.predict(img_array)[0][0])

print("Raw prediction:", prediction)


# IMPORTANT:
# fire = 0
# no_fire = 1

if prediction < 0.5:
    print("🔥 Fire Detected")
else:
    print("✅ No Fire")

print("Confidence:", prediction)