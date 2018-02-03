import csv
import cv2
import numpy as np

lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    steering_center = float(line[3])
    
    correction = 0.2
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    path = '../data/'
    img_center = cv2.imread(path + line[0].strip())
    # img_left = cv2.imread(path + line[1].strip())
    # img_right = cv2.imread(path + line[2].strip())

    images.append(img_center) 
    # images.append(img_left)
    # images.append(img_right)

    measurements.append(steering_center)
    measurements.append(steering_left)
    measurements.append(steering_right)

    # measurements += [steering_center, steering_left, steering_right]

    # source_path = line[0]
    # filename = source_path.split('/')[-1]
    # current_path = '../data/IMG/' + filename
    # image = cv2.imread(current_path)
    # images.append(image)

    # measurement = float(line[3])
    # measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((70, 25), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save('model.h5')