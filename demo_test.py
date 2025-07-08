import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import random

# Define class labels
class_labels = {0: 'Melanocytic_Nevi', 1: 'Normal_Skin'}

# Image dimensions
IMG_HEIGHT = 224
IMG_WIDTH = 224

def load_demo_model():
    """
    Create a dummy model for demonstration purposes.
    This will be replaced with your actual trained model.
    """
    print("Demo mode: Using random predictions")
    return "demo_model"

def preprocess_image(image_path, img_height, img_width):
    """
    Preprocess the input image for model prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image_demo(model, image_path):
    """
    Demo prediction function that generates random predictions.
    """
    # Preprocess the image
    img_array = preprocess_image(image_path, IMG_HEIGHT, IMG_WIDTH)
    
    # Generate random prediction for demo
    prediction = random.uniform(0.1, 0.9)
    
    # Determine class and confidence
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction
    
    # Display the result
    print(f"Predicted Class: {class_labels[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")
    
    # Plot the image with the prediction
    img = load_img(image_path)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_labels[predicted_class]} ({confidence:.2f})")
    plt.show()
    
    return predicted_class, confidence

def main():
    """
    Main function to test the demo with an input image.
    """
    # Load demo model
    model = load_demo_model()
    
    # Path to the test image
    image_path = "download.jpg"
    
    # Perform prediction
    predict_image_demo(model, image_path)

if __name__ == "__main__":
    main()
