# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/flipped.jpg "Flipped"
[image3]: ./examples/left.jpg "Left"
[image4]: ./examples/right.jpg "Left"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (only changed the speed)
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a modified version of the [nVidia pipeline] (https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png).

The model consists of normalization layer, followed by 5 convolution layers, then 4 fully connected layers with dropout between. (detailed further below)

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each of the fully connected layers in order to reduce overfitting (clone.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 14-15), splitting the provided data set 80% test, 20% test. Finally, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I used 4 Epochs after noticing the validation loss was plateauing / rising while the test loss decreased.

```
25712/25712 [==============================] - 47s - loss: 0.2105 - val_loss: 0.0933
Epoch 2/5
25712/25712 [==============================] - 45s - loss: 0.1714 - val_loss: 0.0962
Epoch 3/5
25712/25712 [==============================] - 45s - loss: 0.1584 - val_loss: 0.1056
Epoch 4/5
25712/25712 [==============================] - 45s - loss: 0.1488 - val_loss: 0.0956
Epoch 5/5
25712/25712 [==============================] - 45s - loss: 0.1403 - val_loss: 0.0964
```

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (clone.py line 100).

#### 4. Appropriate training data

I used the training data provided because I couldn't get fine-grained control with my keyboard. I augmented the training to avoid over steering to the left and to provide guidance how to correct upon veering.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model, focus on some pre-processing steps, then upgrade to a more advanced model, combat overfitting and finally augment the test data provided.

My first step was to use a convolution neural network model similar to the leNet architecture predicting the continuous variable of the steering wheel instead of a categorical variable. I thought this model might be appropriate because its a simple enough starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training (80%) and validation set (20%). When training I used mean squared error as my loss function.

Next I added basic preposessing steps of normalization and image cropping (only keeping the relevant road from the scene). 

Then I upgrade to the nVidia based model since its much more powerful.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Preprocess: 
1) Normalize
2) Cropping (remove the top 70 rows to get rid of the horizon, and bottom 25 rows to get rid of the front of the car, only leaving the relevant road), leave a 65x320 image

The nVidia model, with 50% dropout layers between the fully connected layers.
3) 2D Convolution 24@31x158 with 2,2 subsampling/stride
4) 2D Convolution 36@14x77 with 2,2 subsampling/stride
5) 2D Convolution 48@5x37 with 2,2 subsampling/stride
6) 2D Convolution 64@3x35 
7) 2D Convolution 64@1x33
8) Flatten
9) 100 node fully connected layer, w/ drop out and RELU activation
10) 50 node fully connected layer w/ dropout and RELU activation
11) 10 node fully connected layer 2/dropout and RELU activation


```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 10)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dropout_3[0][0]                  
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

I used the provided training data and augmented, by adding a LR flip of the center camera image with negative version of the steering since the training data was biased toward turning one direction (left).

Center: Steering 0.0

![alt text][image1]

Flipped: Steering -0.0 (i know its still zero in this case)

![alt text][image2]

I also noticed that if the car started to veer that it hadn't been trained to recover properly, so I took the left and right camera images and added and subtracted a correction factor to bring the car back to the center. I ended up using a relatively high correction factor (0.8) to keep the car on the track especially on one segment (pictured below) where the dirt boundary confused the model initially. Unfortunately, this large correction causes the car to veer a little back and forth vs being very smooth.

Left: Steering 0.0 + 0.8 = 0.8

![alt text][image3]

Right: Steering 0.0 - 0.8 = -0.8

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the validation plateauing. I used an adam optimizer so that manually training the learning rate wasn't necessary. I also used generators to make sure only the data needed was in memory.
