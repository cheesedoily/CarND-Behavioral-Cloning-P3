import csv
import cv2
import numpy as np
import sklearn


lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def load_image(filename, path='../data/'):
    full_file_path = path + filename.strip()
    image = cv2.imread(full_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def generator(samples, batch_size=32, augment=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                steering_center = float(line[3])

                correction = 0.8
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                img_center = load_image(batch_sample[0])
                img_left = load_image(batch_sample[1])
                img_right = load_image(batch_sample[2])

                images.append(img_center)
                angles.append(steering_center)

                if augment:
                    images.append(cv2.flip(img_center, 1))
                    angles.append(steering_center * -1.0)

                    images.append(img_left)
                    angles.append(steering_left)
                    # images.append(cv2.flip(img_left, 1))
                    # angles.append(steering_left * -1.0)


                    images.append(img_right)
                    angles.append(steering_right)
                    # images.append(cv2.flip(img_right, 1))
                    # angles.append(steering_right * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, 
                                augment=False) # do not need to augment validation set


from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Cropping2D

def make_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0)),))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

model = make_model()

model.summary()

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
                    samples_per_epoch=len(train_samples) * 4,
                    validation_data=validation_generator, 
                    nb_val_samples=len(validation_samples), 
                    nb_epoch=4, 
                    verbose=1)
model.save('model.h5')