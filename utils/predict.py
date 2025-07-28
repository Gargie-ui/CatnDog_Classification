import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import load_model

# Load your old model
model = load_model("model/cat_dog_neither_classifier_mobilenet.h5", compile=False)

# Re-save it in a cleaner .h5 format
model.save("model/cat_dog_neither_classifier_cleaned.h5", save_format='h5')


# Class names — must match the order used during training
class_names = ['cat', 'dog', 'neither']

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # ✅ Match model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)[0]

    prediction /= np.sum(prediction)  # Normalize
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[class_index], round(confidence * 100, 2)

if __name__ == "__main__":
    image_path = "dog.webp"  # You can replace this with a path from CLI
    label, confidence = predict_image(image_path)
    print(f"Prediction: {label} ({confidence}%)")
