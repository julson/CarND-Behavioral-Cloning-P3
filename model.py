import csv
import os
from sklearn.model_selection import train_test_split
from random import shuffle
import cv2
import numpy as np
import sklearn
import argparse

samples = []
with open('./data/driving_log.csv') as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def load_batch(batch_samples):
    images = []
    angles = []
    for batch_sample in batch_samples:
        convert_paths = lambda x : './data/IMG/' + x.split('/')[-1]
        cam_imgs = [cv2.imread(convert_paths(batch_sample[i])) for i in range(3)]

        steering_center = float(batch_sample[3])
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        steering_angles = (steering_center, steering_left, steering_right)

        images.extend(cam_imgs)
        angles.extend(steering_angles)

        cam_imgs_flipped = [np.fliplr(img) for img in cam_imgs]
        steering_angles_flipped = [-angle for angle in steering_angles]

        images.extend(cam_imgs_flipped)
        angles.extend(steering_angles_flipped)


    return images, angles

def generator(samples, batch_size=32):
    n_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset : offset+batch_size]
            images, angles = load_batch(batch_samples)

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D

parser = argparse.ArgumentParser(description='Train model with driving data')
parser.add_argument('model', default=None, nargs='?')
args = parser.parse_args()

if args.model and os.path.isfile(args.model):
    print('Loading', args.model)
    model = load_model(args.model)
else:
    print('Creating model from scratch')
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

model.fit_generator(train_generator, samples_per_epoch=len(train_samples), \
                    validation_data=validation_generator, nb_val_samples=len(validation_samples), \
                    nb_epoch=5)

model.save('model.h5')
