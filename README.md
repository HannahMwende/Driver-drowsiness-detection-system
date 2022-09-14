# Driver-Drowniness-dectection-system

<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs13177-019-00199-w/MediaObjects/13177_2019_199_Fig3_HTML.png" width="1000" height="500">

Authors:

* Ann Maureen
* Hannah Mutua
* Ibrahim Hafiz
* Samuel Kabati
* Angela Cheruto
* Janet Gachoki

## Overview

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving.

The objective of this intermediate Python project is to build a drowsiness detection system that will detect whether a person’s eyes are closed for a few seconds,the system will then alert the driver when drowsiness is detected.

The data obtained from [Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset).It was created with the help of a larger-scale dataset of human eyes from Media Research Lab.

## Project aim

To develop a neural network that can determine if eyes are open or closed in conjunction with computer vision.It will determine whether a driver has had their eyes closed for longer than three seconds and if they have,the driver receives an alert.

Deploy our model to a webcam application that will classify a driver as having slept on the wheel if the positive class(eyes closed) is true for more than three frames.

## Data

This project uses data from [Kaggle](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset), which can also be found in the data folder in this project's GitHub repository. The data contains 2000 open eyes and 2000 closed eyes.The dataset contains infrared images in low and high resolution, all captured in various lightning conditions and by different devices. 

The dataset has no class imbalance with open eyes and closed eyes having a ration of 1:1.

## Defining Experimental design

* Importing the relevant libraries used in the analysis.

* Load and preview images

* Explore the dataset we will use for our project.

* Exploratory Data Analysis (EDA)

* Data Pre-processing

* Modelling and Evaluation

* Challenging the model

* Conclusion

* Recommendations


### Libraries

Driver drowsiness detection is a project built using OpenCV with Python as a backend language.

Tensorflow

Matplotlib

Seaborn

Scikit-Learn


## Data exploration

![Screenshot (357)](https://user-images.githubusercontent.com/104419035/190087849-83464533-3577-4fe0-b5e7-7e18d41c42dd.png)

The image represents the count of open eyes and closed eyes in our dataset.

We Loaded and previewed images with their corresponding labels as shown.

![Screenshot (356)](https://user-images.githubusercontent.com/104419035/190087759-5d914e07-20a7-4199-aa9e-4af2aaf0c58b.png)

The images were classified as either open or closed.



## Data pre-processing 

The datasets containing the open and closed eyes were combined.

The combined data was then split into X and y, with X being images and y being labels.

Reshaping of data was done to ensure images are of a fixed size

Normalization of data was done to reduce pixel values to a range between 0 and 1 by dividing the values by 255 to enable faster computation.



## Modelling and Evaluation

After building the model, we did several GridSearch to find optimum values for hyperparameter. 
The optimal parameters are :

dense_neurons1: 256
dense_neurons2 : 512
dense_neurons3 : 512
dropout : 0.1
epochs : 10
filters : 32
layout: 2*3x3
pooling : 1

From the grid search we find the best layout to be 2*3x3 with 32 filters. This means we will have two convolution layers with a kernel size of (3,3) representing the height and width of the filter. 

#### Final model evaluation

![Screenshot (361)](https://user-images.githubusercontent.com/104419035/190208473-66697394-0aa8-46aa-ac37-a2e59ecd05d8.png)

The final model had a Training auc score of 0.99 and Test auc score of 0.99

![Screenshot (362)](https://user-images.githubusercontent.com/104419035/190209468-87377507-72ee-4f63-992e-f13ef660d355.png)

The plot shows the loss and auc curves for training and validation data.


A confusion matrix for our final model was plotted as seen below.

![Screenshot (364)](https://user-images.githubusercontent.com/104419035/190210056-15374134-bcc5-4e66-9e97-54a49e64e11c.png)


The model resulted in a Precision score of 0.99,Recall score of 1.0,Accuracy score of 1.0 and F1 score 0f 0.99

## Challenging the model

A transfer learning model was used to challenge the model.Transfer learning for image classification is about leveraging feature representations from a pre-trained model, so you don't have to train a new model from scratch.VGG19 was used as it is an advanced network with pretrained layers and a better understanding of what defines an image in terms of shape, color, and structure.

![Screenshot (366)](https://user-images.githubusercontent.com/104419035/190210284-56650542-efcd-482c-ac81-7fc1b52d96e1.png)

The image above shows the training auc score and test auc score of the transfer learning model and the plot below shows loss and auc curves for training and validation data of the transfer learning model.

![Screenshot (365)](https://user-images.githubusercontent.com/104419035/190210581-3b76ac6f-92f6-4563-a564-1ccfa1eb367a.png)


## More Information
For detailed information kindly refer to our Notebook and presentation slides found in the [Github repository](https://github.com/HannahMwende/Driver-Drowniness-dectection-system).
