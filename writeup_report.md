# **Behavioral Cloning Project**

## Goals
* Use the simulator to collect data of good driving behavior
* Build, a convolutional neural network using Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[center]: ./images/center_cam.jpg "Center Camera Image"
[left]: ./images/left_cam.jpg "Left Camera Image"
[right]: ./images/right_cam.jpg "Right Camera Image"
[recovery1]: ./images/recovery1.jpg "Recovery Image 1"
[recovery2]: ./images/recovery2.jpg "Recovery Image 2"
[recovery3]: ./images/recovery3.jpg "Recovery Image 3"

## Files Included
* model.py - creates and trains the model
* drive.py - drives the car in autonomous mode
* model.h5 - contains a trained convolutional neural network that can run on track 1
* video.mp4 - video of a successful autonomous run on track 1
* writeup_report.md - report summary
* model_track2.h5 - contains a trained convolutional network that can run on track 2
* video_track2.mp4 - video of a successful autonomous run on track 2

## Model Architecture

The architecture uses a slightly modified version of Nvidia's PilotNet architecture. PilotNet has been used to successfully predict steering angles in real-world conditions, so I figured it's a good enough architecture to start with. The architecture uses 5 convolutional layers with 3x3 and 5x5 filters and 4 fully-connected layers (one less than PilotNet's). Each layer uses RELU for activation. Some minor preprocessing steps were used, like cropping the image to eliminate the area above the horizon and the hood of the car and normalizing the image values. The learning rate tuning was needed due to the usage of the Adam optimizer. Five epochs were used as I wasn't getting significant loss reduction after that (although it seems like 3 or 4 would have been enough too).

Here is the Keras implementation of the model:
```python
model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.5 - 0.5))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(Flatten())

model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
```

## Design and Training Strategy

I mainly used data augmentation right from the start to avoid overfitting. In this case, I took the left and right camera images, mapped them to the center steering angle with a correction offset (+/- 0.2) and added them as part of the training set. I also flipped the images, providing a negated steering angle for each. To avoid creating a separate validation set, I split 20% of the sample data to be used for validation.

Another way that I found that helped in shortening the training time is the train the model in increments, keeping successful models and using them as the foundation for the additonal data that I've gathered.

## Creating the Training Set (only Track 1)

I recorded 6 laps of center lane driving going both directions. Here's an example from from one time instance (all 3 cameras):

![alt text][left]
![alt text][center]
![alt text][right]

Recovery maneuvers were also recorded just in case a vehicle veers off the lane:

![alt text][recovery1]
![alt text][recovery2]
![alt text][recovery3]

Other techniques included driving while hugging the edges (especially for turns). The initial data gathering phase produced some inconsistent results, so I had to go in and record more recovery data to iron out the kinks.

Overall, I ended up with a good model from 41412 images from track 1. I eventually tried to augment the track 1 data with my training data from track 2, which failed to create a general model for both tracks, but that's for another story (more below).

## Results

The model produced satisfactory results, with the car being able to run autonomously along track 1 without veering off-course. There is one turn where the car manages to go past the lane markings, but it quickly recovers itself. The model is far from perfect though. If you run the simulation long enough, it eventually gets stuck on something (especially that aforementioned turn).

## Improvements for the Future

More training data, especially recovery data, would probably help in making the model smarter in navigating turns. I haven't included any dropout in the model, but it's something that I can play around in the future too. And of course, successfully incorporating track 2 data would help in making the model more robust

## The Case of Track 2

In an effort to create a generalized model, I eventually gathered track 2 data which I added to my successful model. I used the same methods for creating the training data, including recovery maneuvers and being able to drive on the opposite lane (which is probably not a safe idea :D). The eventual model was successful in navigating track 2 in a completely unsafe way (see video_track2.mp4), but at the expense of track 1 performance (veers off-course this time). I am still in the process of fixing it, but for the purposes of this writeup, it will not make it in time.
