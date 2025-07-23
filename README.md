#Melanocytic Nevi Classification Using Deep Learning

## Abstract

Melanocytic nevi, one of the most common skin diseases, require accurate and early diagnosis to ensure timely treatment. Traditional diagnostic methods—visual inspections and biopsies—are invasive, time-consuming, and prone to human error. To address these challenges, we propose a deep learning-based system for efficient and automated classification of skin conditions.

Leveraging the ResNet50 architecture with transfer learning, our system incorporates advanced preprocessing, data augmentation, and custom layers optimized for binary classification. With its robust design and non-invasive approach, this system offers a scalable and cost-effective solution for real-time skin disease diagnosis, bridging the gap between traditional and AI-driven healthcare solutions.

---

## Model Architecture

![Model Architecture](https://github.com/user-attachments/assets/88c6b07b-a9fc-4a3b-ad63-77b8a901fb36)

---

## Methodology

Our methodology comprises the following key steps:

1. **Preprocessing for Dataset Augmentation and Balancing**
2. **Feature Extraction using Transfer Learning with ResNet50**
3. **Classification using a Fully Connected Neural Network**

### 1. Preprocessing and Augmentation

To ensure effective training, the raw dataset was preprocessed by resizing all images to a resolution of 224x224 pixels, standardizing the input size for the ResNet50 model. Data augmentation techniques—including rotation, flipping, and zooming—were applied to introduce variability and mitigate overfitting.

Class imbalance was addressed by calculating class weights and applying them during model training.

![Data Augmentation](https://github.com/user-attachments/assets/e207db2c-270d-48be-9422-90524e8eb45d)

### 2. Transfer Learning for Feature Extraction

We employed the ResNet50 architecture, pre-trained on the ImageNet dataset, as the feature extractor. The feature maps from the ResNet50 output were passed through custom layers for binary classification (see Model Architecture above).

The feature extraction is mathematically defined as:

```
f(x) = ReLU(W.x + b)       (1)
```
- **f(x)**: Activation function
- **W, b**: Weights and biases of the layer
- **x**: Input feature vector

### 3. Classification

The final classification layer uses a fully connected neuron with a sigmoid activation function, defined as:

```
ŷ = 1 / (1 + e^(-z))       (2)
```
where:
- **z = W.x + b**
- **W, b**: Weights and biases
- **x**: Feature vector

The model is trained using binary cross-entropy loss:

```
L = -1/N ∑[y_i log(ŷ_i) + (1 - y_i) log(1 - ŷ_i)]       (3)
```
- **y_i**: True labels
- **ŷ_i**: Predicted probabilities
- **N**: Number of samples

---

## Result Analysis: Confusion Matrix

The confusion matrix provides an in-depth evaluation of the classification model's performance:

- **True Positives (Melanocytic Nevi):** 1195 samples correctly identified.
- **True Negatives (Normal Skin):** 586 samples correctly classified.
- **False Positives:** Only 1 normal skin sample misclassified as melanocytic nevi.
- **False Negatives:** 0 cases; the model consistently identified all melanocytic nevi.

This outstanding performance reflects high precision, recall, and reliability in skin lesion classification tasks.

![Confusion Matrix](https://github.com/user-attachments/assets/5ff82ae1-7daa-4ba3-af7a-2c38a4a557d8)

---

## Conclusion

The ultimate research objective is to build a state-of-the-art application for non-invasive melanocytic nevi detection using computer vision and image processing. This solution aspires to become a global standard, offering a less expensive and more accessible alternative to traditional diagnostic systems.

---

## License

This project is licensed under the [MIT License](./LICENSE).
