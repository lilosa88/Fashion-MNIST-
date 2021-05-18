# Fashion-MNIST

# Objective

- This project belongs to [kaggle's competitions](https://www.kaggle.com/zalando-research/fashionmnist) and I carried out as a part of a specialization called [DeepLearning.AI TensorFlow Developer Specialization](https://www.coursera.org/account/accomplishments/specialization/certificate/L6R6AFWVXHZT) which is given by DeepLearning.AI. This specialization is conformed by 4 courses: 
1. Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning 
2. Convolutional Neural Networks in TensorFlow 
3. Natural Language Processing in TensorFlow 
4. Sequences, Time Series and Prediction

Specifically this project is part of the first course in this specialization. 

- MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

- Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

- The objective of this study is to correctly identify the different Zalando's articles from a dataset of tens of thousands of handwritten images.

# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, seaborn, matplotlib

# Data description  

- Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

- The training and test data sets have 785 columns. The first column consists of the class labels (see above), and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.

- Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot

- Each row is a separate image, where: Column 1 is the class label and the remaining columns are pixel numbers (784 total). Each value is the darkness of the pixel (1 to 255)

- The figures looks like that
  <p align="center">
   <img src="https://github.com/lilosa88/Fashion-MNIST-/blob/main/Images/Captura%20de%20Pantalla%202021-05-18%20a%20la(s)%2016.20.14.png" width="160" height="140">
  </p> 
  
# Feature engineering
  
- Defining X and Y from the df_train dataset
- We check which is the maximum value that you can find in one row of the df_train dataset. The maximum value is 255. If we are training a neural network, for    various reasons it's easier if we treat all values as between 0 and 1, therefore we need to normalize our datasets.
- So we normalize dividing by 255.
- Resahping, following X = X.values.reshape(-1, 28,28,1)
- Label encoding for the y label
- Split into train and test

# Neural Network model

We compare the performance of two following two neural networks:
- Simple Model (Accuracy 0.97238)
- Model with double convolutions and pooling (Accuracy 0.9864)

In both case the activation functions used were 'relu' and 'softmax', the lr = 0.001 and as loss function we use categorical_crossentropy.
