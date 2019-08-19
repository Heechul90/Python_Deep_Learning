### Dogs vs Cats
# Kaggle Dataset의 일부를 이용한 개, 고양이 구분
# Dog Image: 1,111개, Cat Image: 1,111개, 총 2,222개
# 출처: pontoregende GitHub

# 필요한 라이브러리
import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split


path = 'Python_Deep_Learning/Dogs_vs_Cats/train/'
# Data dimensions and paths
img_width = 150
img_height = 150
TRAIN_DIR = './Dogs_vs_Cats/train/'
TEST_DIR = './Dogs_vs_Cats/test1/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


# Helper function to sort the image files based on the numeric value in each file name.
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

# Sort the traning set.
# Use 1300 images each of cats and dogs instead of all 25000 to speed up the learning process.
# Sort the test set

train_images_dogs_cats.sort(key=natural_keys)
train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13800]

test_images_dogs_cats.sort(key=natural_keys)

# Now the images have to be represented in numbers.
# For this, using the openCV library read and resize the image.
# Generate labels for the supervised learning set.
# Below is the helper function to do so.

def prepare_data(list_of_images):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    x = []  # images as arrays
    y = []  # labels

    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width, img_height),
                            interpolation=cv2.INTER_CUBIC))

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
        # else:
        # print('neither cat nor dog name present in images')

    return x, y

# Generate X and Y using the helper function above
# Since K.image_data_format() is channel_last, input_shape to the first keras layer will be (img_width, img_height, 3).
# '3' since it is a color image

X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())

# Split the data set containing 2600 images into 2 parts, training set and validation set.
# Later, you will see that accuracy and loss on the validation set will also be reported while fitting the model using training set.

# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

# We will be using the Sequential model from Keras to form the Neural Network.
# Sequential Model is used to construct simple models with linear stack of layers.

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Prepare generators for training and validation sets

train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)
validation_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)

# Start training the model!

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size
)