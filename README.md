---ABSTRACT---
One of the commonest skin disease Melanocytic nevi requires accurate and early diagnosis to ensure timely treatment. 
Traditional diagnostic methods, which rely on visual inspections and biopsies, are invasive,
time-consuming, and prone to human error. To address these challenges, we propose a deep learning-based system for efficient and automated classification of skin con-ditions. 
Leveraging the ResNet50 architecture with transfer learning, our sys-tem incorporates advanced preprocessing, 
data augmentation, and custom layers optimized for binary classification. With its robust design and non-invasive approach,
this system offers a scalable and cost-effective solution for real-time skin disease diagnosis, bridging the gap between traditional and AI-driven healthcare solutions.

------------------MODEL ARCHITECTURE--------------------

![image](https://github.com/user-attachments/assets/88c6b07b-a9fc-4a3b-ad63-77b8a901fb36)

-------METHODOLOGY-------
The methodology adopted in this study consists of the following key steps:
1.	Preprocessing for dataset augmentation and balancing.
2.	Feature extraction using transfer learning with ResNet50.
3.	Classification using a fully connected neural network

1.preprocessing and augmentation--->
----------	Preprocessing and Augmentation  ------------
To ensure effective training, the raw dataset was preprocessed by resizing all images to a resolution of 224x224 pixels,
standardizing the input size for the ResNet50 model. Data augmentation techniques, such as rotation, flipping, and zooming,
were applied as shown in Fig. 2. to introduce variability and mitigate overfitting. Class imbalance was addressed by calculating class weights and ap-plying them during model training.
![image](https://github.com/user-attachments/assets/e207db2c-270d-48be-9422-90524e8eb45d)


2. Transfer Learning for Feature Extraction--->
We employed the ResNet50 architecture, pre-trained on the ImageNet
dataset, as the feature extractor. The feature maps from the ResNet50 output
were passed through additional custom layers for binary classification as shown in the model architecture in Fig. 3. The model uses the following equation for feature extraction given in Eq(1):

	f(x)= ReLU(W.x+b)	(1)
where: 
f(x) = activation function, 
W, b = represents the weights and biases of the layer,
 x = input feature vector.

4. Classification--->
The final classification layer uses a fully connected dense neuron with a sigmoid activation function, given by	Eq (2):
                                                                       y ̂=1/(1+e^(-z) )                                                           (2)
  
where:
		z = W. x + b, 
	    W, b = represents the weights and biases
		x= feature vector
The model was trained using binary cross-entropy loss, defined as Eq (3):
	                                    L=-1/N Σ_(i=1)^N [y_i  log⁡(y ̂_i )+(1-y_i )  log⁡(1-y ̂_i )]                   (3)     
where:
		yi = true labels.
		y ̂_i= predicts the probabilities.
		N= number of samples. 



RESULT ANALYSIS using confusion matrix---->
The confusion matrix provides an in-depth evaluation of the classification mod-el's performance. True positives (melanocytic nevi) stand out prominently, with 1195 samples correctly identified, highlighting the model's strong ability to de-tect melanocytic nevi accurately. Similarly, the true negatives (normal skin) showcase 586 correctly classified samples, demonstrating excellent performance in identifying normal skin. The model exhibits minimal error, with only one false positive, where a normal skin sample was misclassified as melanocytic nevi. Re-markably, there are no false negatives, indicating the model consistently identi-fied all melanocytic nevi without overlooking any cases. This performance re-flects the model's high precision, recall, and reliability in skin lesion classifica-tion tasks.
![image](https://github.com/user-attachments/assets/5ff82ae1-7daa-4ba3-af7a-2c38a4a557d8)



conclusion---->
The ultimate research objective is to build a state-of-the-art application for non-invasive melanocytic nevi detection using computer vision and image processing,
which would stand to be a global standard and be less expensive as compared to other traditional systems which are already in place for the same
