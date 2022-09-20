# Driver-Drowniness-detection-system

<img src="https://media.springernature.com/lw685/springer-static/image/art%3A10.1007%2Fs13177-019-00199-w/MediaObjects/13177_2019_199_Fig3_HTML.png" width="1000" height="600">

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

The optimal parameters were:
Units : 128,
Dropout : 0.1,
Epochs: 10, 
Filters : 32, 
Layout : 3*3,
Pooling: None

From the grid search we find the best layout to be 3x3 with 32 filters. This means we will have a convolution layer with a kernel size of (3,3) representing the height and width of the filter. 

#### Final model evaluation

![Screenshot (381)](https://user-images.githubusercontent.com/104419035/190605635-898b3792-3210-4a97-8aeb-1ba4c83efa58.png)

The final model had a Training auc score of 0.99 and Test auc score of 0.99

![Screenshot (380)](https://user-images.githubusercontent.com/104419035/190605421-07862721-886d-4d99-8ec8-e4eedc34af00.png)

The plot shows the loss and auc curves for training and validation data.


A confusion matrix for our final model was plotted as seen below.

![Screenshot (367)](https://user-images.githubusercontent.com/104419035/190600080-4a8a6874-2b0d-4f4f-adfc-36406e71ca1f.png)


The model resulted in a Precision score of 0.97,Recall score of 1.0,Accuracy score of 0.99 and F1 score 0f 0.99

## Challenging the model

A transfer learning model was used to challenge the model.Transfer learning for image classification is about leveraging feature representations from a pre-trained model, so you don't have to train a new model from scratch.VGG19 was used as it is an advanced network with pretrained layers and a better understanding of what defines an image in terms of shape, color, and structure.

![Screenshot (379)](https://user-images.githubusercontent.com/104419035/190604928-9dc66c1e-42e1-4dde-9cb4-8ea220b34095.png)

The image above shows the Training auc score of 0.99 and Test auc score of 0.99 of the transfer learning model and the plot below shows loss and auc curves for training and validation data of the transfer learning model.

![Screenshot (378)](https://user-images.githubusercontent.com/104419035/190604614-5ebb1578-619f-4dec-a58e-0a2d1c6b32d4.png)

## Conclusion
The final model achieved an auc_score of 0.99 which surpassed the earlier set target of 0.97 meaning that the model is able to separate open and closed eyes 99 out of a hundred.

The model was deployed on a webcam application where a video of the driver’s eyes is captured within 3 video frames and once closed eyes are detected it sends out an alert to the driver.

Upon successful deployment of the model, the number of road accidents can be reduced when this system is implemented in the vehicle to detect the drowsiness of the driver.

## Recommendations
Given the success of this driver drowsiness detection system with Dereva automobile company,other car companies,bodies and authorities involved in road safety such as the National Transport and Safety and Authority can adopt the system.

## Future scope
In future works,the system can be improved significantly by using other signs of drowsiness like blinking rate, yawning, drifting from one’s lane so as to ensure the system is reliable in detecting drowsiness. If all these parameters are used it can improve the accuracy.

## More Information
For detailed information kindly refer to our Notebook and presentation slides found in the [Github repository](https://github.com/HannahMwende/Driver-Drowniness-dectection-system).
