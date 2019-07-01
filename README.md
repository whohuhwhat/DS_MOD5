# Convolutional Neural Network Image Classification: Emotions, Race, Gender

# Project Overview
For my final project, I took a deeper dive into Convolutional Neural Networks (CNN).  Using Keras, TensorFlow GPU and cuDNN, I trained a CNN to take an image and have it classify what emotion the person in the image was expressing.  I then trained additional CNNs to classify the race and gender of the person in the image.  Here are a few examples of my results:

<img width="1056" alt="predictions" src="https://user-images.githubusercontent.com/30739929/60197625-b890b980-980d-11e9-8c04-b62657b4fda5.png">


**Data Stack**
* Python
* Keras
* TensorFlow GPU
* cuDNN
* Numpy
* Seaborn
* Pandas

# Training Data/EDA
The **fer2013 dataset**, which consisted of 35,000 images with 7 different classes, was used to train the emotion CNN.  The 7 different emotions classified were Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral. The images were 48x48 in size.  
<p align="center">
<img width="703" alt="fer2013_train" src="https://user-images.githubusercontent.com/30739929/60436975-44725f00-9bdb-11e9-9aae-38a6681df8d0.png"></p>

There was a class imbalance with Happy images which contained over 7000 images.  Disgust images had less than 1000 images.  The distribution of the rest of the emotions were fairly balanced at around 3000 to 4000 images.

<p align="center">
<img alt="Training Data Distribution" src="https://user-images.githubusercontent.com/30739929/60437508-8354e480-9bdc-11e9-9ab1-838c839f710f.png">
</p>

The **Wilma Bainbridge 10k Faces dataset** was used to classify race and gender.  Gender was classified as male or female.  Race was classified as Other, Caucasian, Black, East Asian, South Asian, Hispanic, and Middle Eastern.
<p align="center">
<img width="915" alt="wm" src="https://user-images.githubusercontent.com/30739929/60437877-489f7c00-9bdd-11e9-8015-b7ffd1f8edf8.png"></p>

## Data Augmentation
Data Augmentation was used to flip, shear and rotate images.  This gives the CNN more data to train on and gives it more possible real world scenarios for feature extraction.
<p align="center">
<img width="763" alt="data_augmentation" src="https://user-images.githubusercontent.com/30739929/60438231-38d46780-9bde-11e9-94c6-4ce803529915.png"></p>

## Visiualizing Different Layers in the CNN
Keras was used to visualize different layers in the CNN.  
<p align="center">
  <img width="627" alt="layers" src="https://user-images.githubusercontent.com/30739929/60438907-bba9f200-9bdf-11e9-8a01-549056c47cb0.png"></p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/30739929/60439071-13485d80-9be0-11e9-8d01-2c8a52b4a455.png"></p>
  
# Models
Validation Accuracy was used as a metric to see how well the CNN performed.

**Basic Model Architecture:**
* Conv2D
* Batch Normalization
* MaxPooling
* Dropout (0.5)
* Dense Activation: softmax

**Deep Learning Architecture:**
* Conv2D
* Activation: Relu
* Batch Normalization
* Conv2D
* Activation: Relu
* Batch Normalization
* MaxPooling
* Dropout(0.5)
* Flatten

The best validation accuracy achieved was 0.6640

![66plot](https://user-images.githubusercontent.com/30739929/60440102-10e70300-9be2-11e9-8b80-11f57019de55.png)
![66loss](https://user-images.githubusercontent.com/30739929/60440100-10e70300-9be2-11e9-972f-42acb944f784.png)


# Transfer Learning
Transfer Learning is the use of a pretrained network that was previously ran on a large, general data set and saved.  The hierarchical features learned by this network can act as a generic model and can be used for different computer vision tasks including classifying completely different classes of images. Transfer Learning was done using Keras' pretrained models.  Xception, VGG16 and InceptionV3 were used.


I trained a model using the Xception pretrained model with varying frozen layers. In the first model, the Xception pretrained model was not trainable and ran 

* Xception trainable=False
* Dropout
* Flatten
* Dense 132, activation=relu
* Dense 7, activation=softmax

This model produced a validation accuracy: 0.3576.

Looking to improve upon this I Unfroze the last block of layers of Xception which consisted of 6 layers.  This produced a validation accuracy: 0.5460.  I saw a major improvement from my previous model so I went on to unfreeze more layers of Xception.  With blocks 12-14 unfrozen and trainable, the validation accuracy was 0.6017.

Finally I trained the model with just the first layer of Xception frozen and untrainable.  This yielded my best validation accuracy of 0.6641.  Unfortunately this is only a 0.0001 increase from my model without transfer learning.


![xcep_acc](https://user-images.githubusercontent.com/30739929/60440135-26f4c380-9be2-11e9-9665-9519b56ec1f2.png)
![xcep_loss](https://user-images.githubusercontent.com/30739929/60440136-26f4c380-9be2-11e9-95cf-934f283e225c.png)

# Results
I combined my three different convolutional neural networks, one for emotion, another for race, and finally, one for gender.  These were my results:

<img width="1056" alt="predictions" src="https://user-images.githubusercontent.com/30739929/60197625-b890b980-980d-11e9-8c04-b62657b4fda5.png">

Unfortunately, the dataset I used to classify race was not diverse and had a very large class imbalance.  Nearly 80% of the images were labeled as caucasian.  This resulted in plenty of misclassifications.

<img width="1082" alt="misclassified_results" src="https://user-images.githubusercontent.com/30739929/60445353-0bdb8100-9bed-11e9-8c32-637a698b95ee.png">

**A neural network is only as good as your training data**, and thus, my model had a very difficult time distinguishing non-caucasians.  I need a more diverse dataset to accurately classify race.

# Next Steps
1. Use a more diverse dataset!!  
2. Expand Live video demo to use all three CNNs to classify emotion, race, and gender.
3. I used OpenCV to detect faces. I would like to use other facial detection models such as MTCNN.
