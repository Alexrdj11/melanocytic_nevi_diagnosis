One of the commonest skin disease Melanocytic nevi requires accurate and early diagnosis to ensure timely treatment. 
Traditional diagnostic methods, which rely on visual inspections and biopsies, are invasive,
time-consuming, and prone to human error. To address these challenges, we propose a deep learning-based system for efficient and automated classification of skin con-ditions. 
Leveraging the ResNet50 architecture with transfer learning, our sys-tem incorporates advanced preprocessing, 
data augmentation, and custom layers optimized for binary classification. With its robust design and non-invasive approach,
this system offers a scalable and cost-effective solution for real-time skin disease diagnosis, bridging the gap between traditional and AI-driven healthcare solutions.

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

