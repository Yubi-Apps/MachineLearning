import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, img_height, img_width):
    """
    Preprocess an image for prediction with the model.
    Resizes and normalizes the image.
    """
    img = load_img(image_path, target_size=(img_height, img_width))  # Resize image
    img_array = img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1] range
    return img_array

def get_most_recent_image(directory):
    """
    Get the most recently added image file from the specified directory.

    Parameters:
        directory (str): Directory to search for image files.

    Returns:
        str: Path to the most recently added image.
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        raise FileNotFoundError("No files found in the directory.")
    most_recent_file = max(files, key=os.path.getmtime)  # Get the file with the most recent modification time
    return most_recent_file

def predict_image_class(image_path, model, class_names, img_height, img_width):
    """
    Predict the class of an image using the model.

    Parameters:
        image_path (str): Path to the image.
        model: Trained TensorFlow model.
        class_names (list): List of class names corresponding to the model's output indices.
        img_height (int): Image height used during training.
        img_width (int): Image width used during training.

    Returns:
        str: Predicted class name.
    """
    # Preprocess the image
    img_array = preprocess_image(image_path, img_height, img_width)

    # Make prediction
    predictions = model.predict(img_array)
    print(f"Raw predictions: {predictions}")  # Debugging line

    # Safeguard if predictions do not return expected shape
    if predictions.ndim == 2:  # For probabilities, shape should be (1, num_classes)
        predicted_class_idx = np.argmax(predictions[0])  # Take first batch
    else:
        raise ValueError(f"Unexpected prediction output shape: {predictions.shape}")

    # Map the index to the class name
    predicted_class = class_names[predicted_class_idx]
    print(f"Predicted class index: {predicted_class_idx}, class: {predicted_class}")  # Debugging line

    return predicted_class


# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Example class names (replace with actual class names from training)
class_names = [
    'SL17 Phthorimaea operculella (Zeller)', 
    'SL15 Myzus persicae (Sulzer)', 
    'SL01 Agrotis ipsilon (Hufnagel)', 
    'SL05 Bemisia tabaci (Gennadius)', 
    'SL10 Epilachna vigintioctopunctata (Fabricius)', 
    'SL03 Aphis gossypii Glover', 
    'SL06 Brachytrypes portentosus Lichtenstein', 
    'SL02 Amrasca devastans (Distant)'
]

# Example image dimensions
img_height = 180  # Replace with actual image height used in training
img_width = 180   # Replace with actual image width used in training

# Get the most recent image from the "uploads" directory
uploads_dir = "uploads"  # Replace with your uploads folder
try:
    image_path = get_most_recent_image(uploads_dir)
    result = predict_image_class(image_path, model, class_names, img_height, img_width)
    print(f"Predicted class: {result}")
except FileNotFoundError as e:
    print(f"Error: {e}")
