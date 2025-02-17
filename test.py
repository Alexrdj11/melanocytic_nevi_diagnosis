import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Define class labels (adjust according to your model's class indices)
class_labels = {0: 'Melanocytic_Nevi', 1: 'Normal_Skin'}

# Image dimensions (should match the training dimensions)
IMG_HEIGHT = 224  # Replace with your model's input height
IMG_WIDTH = 224   # Replace with your model's input width


def load_trained_model(model_path):
    """
    Load the trained model from the given path.
    """
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path, img_height, img_width):
    """
    Preprocess the input image for model prediction.
    """
    # Load the image
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict_image(model, image_path):
    """
    Predict the class of the given image using the trained model.
    """
    # Preprocess the image
    img_array = preprocess_image(image_path, IMG_HEIGHT, IMG_WIDTH)

    # Predict using the model
    # Get the output probability for binary classification
    prediction = model.predict(img_array)[0][0]

    # Determine class and confidence
    predicted_class = 1 if prediction > 0.5 else 0
    confidence = prediction if predicted_class == 1 else 1 - prediction

    # Display the result
    print(f"Predicted Class: {class_labels[predicted_class]}")
    print(f"Confidence: {confidence:.2f}")

    # Plot the image with the prediction
    img = load_img(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {class_labels[predicted_class]} ({confidence:.2f})")
    plt.show()


def main():
    """
    Main function to test the model with an input image.
    """
    # Path to the saved trained model
    model_path = "best_model.keras"  # Replace with the path to your model
    model = load_trained_model(model_path)

    # Path to the test image
    image_path = "download.jpg"

    # Perform prediction
    predict_image(model, image_path)


if __name__ == "__main__":
    main()
