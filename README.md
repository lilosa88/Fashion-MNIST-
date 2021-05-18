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

- The objective of this study is to correctly identify the different Zalando's articles from the dataset.

# Code and Resources Used

- **Phyton Version:** 3.0
- **Packages:** pandas, numpy, sklearn, seaborn, matplotlib, tensorflow, keras.

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

- The figures looks like:
  <p align="center">
   <img src="https://github.com/lilosa88/Fashion-MNIST-/blob/main/Images/Captura%20de%20Pantalla%202021-05-18%20a%20la(s)%2016.20.14.png" width="190" height="180">
  </p> 
  
# Feature engineering

- The Fashion MNIST data is available directly in the tf.keras datasets API. Using load_data we get two sets of two lists, these will be the training and testing values for the graphics that contain the clothing items and their labels.
 
- The values in the number are between 0 and 255. Since we will train a neural network, we need that all values are between 0 and 1. Therefore, we normalize dividing by 255.

- We reshape the images (only for the second model), following training_images.reshape(60000, 28, 28, 1) and test_images.reshape(10000, 28, 28, 1)


# Neural Network model

### First model: Simple Neural Network

- This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - One flatten layer: It turns the images into a 1 dimensional set.
  - Three Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer         consisted in 1024 neurons with relu as an activation function. The second, have 128 neurons and the same activation function. Finally, the thrird had 10 neurons     and softmax as an activation function. Indeed, the number of neurons in the last layer should match the number of classes you are classifying for. In this case     it's the digits 0-9, so there are 10 of them.

- We built this model using Adam optimizer and sparse_categorical_crossentropy as loss function.

- We obtained Accuracy 0.9299 for the train data and Accuracy 0.8923 for the validation data.

### Second model: Neural Network with convolution and pooling

- This model was created using tf.keras.models.Sequential, which defines a SEQUENCE of layers in the neural network. These sequence of layers used were the following:
  - One Convolution layer with a MaxPooling layer which is then designed to compress the image, while maintaining the content of the features that were                 highlighted by the convlution. By specifying (2,2) for the MaxPooling, the effect is to quarter the size of the image.
  - One flatten layer: It turns the images into a 1 dimensional set.
  - Three Dense layers: This adds a layer of neurons. Each layer of neurons has an activation function to tell them what to do. Therefore, the first Dense layer         consisted in 1024 neurons with relu as an activation function. The second, have 128 neurons and the same activation function. Finally, the thrird had 10 neurons     and softmax as an activation function. Indeed, the number of neurons in the last layer should match the number of classes you are classifying for. In this case     it's the digits 0-9, so there are 10 of them.

- We built this model using Adam optimizer and sparse_categorical_crossentropy as loss function.

- The number of epochs=20

- We obtained Accuracy 0.9953 for the train data and Accuracy 0.9147 for the validation data.
