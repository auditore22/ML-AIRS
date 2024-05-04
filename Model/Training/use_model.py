import tensorflow as tf
import numpy as np
from PIL import Image


def predict_image(model, image_path):

    infer = model.signatures['serving_default']
    # Load and preprocess the image
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match the model's expected input
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the loaded model
    prediction = infer(tf.constant(img_array, dtype=tf.float32))['dense_5']
    predicted_class = np.argmax(prediction.numpy(), axis=1)
    if predicted_class == [0]:
        return "It's a dog."
    else:
        return "It's a cat."
