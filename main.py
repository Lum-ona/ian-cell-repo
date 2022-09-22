import os
import cv2
import imghdr
import numpy as np
import tensorboard
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


DATA_DIR = 'cell_images'
image_exts = ['jpg', 'png', 'bmp', 'jpeg']

def eliminate_dodgy_img():
    for image_class in os.listdir(DATA_DIR):
        for image in os.listdir(os.path.join(DATA_DIR, image_class)):
            image_path = os.path.join(DATA_DIR, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    print('IMAGE NOT IN EXT LIST {}'.formart(image_path))
                    os.remove(image_path)
            except Exception as e:
                print('Issue with Image {}'.format(image_path))
# eliminate_dodgy_img()

def load_data():
    data = tf.keras.utils.image_dataset_from_directory('cell_images')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    return data
# load_data()

def preprocess_data():
    data = load_data()
    data = data.map(lambda x,y: (x/255,y))
    scaled_iterator = data.as_numpy_iterator().next()

    train_size = int(len(data)*.7)+1
    val_size = int(len(data)* .2)
    test_size = int(len(data)* .1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    return train, val
# preprocess_data()

def net():
    model = Sequential()

    model.add(Conv2D(16,(3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3,3), 1, activation="relu"))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # model.summary()
    return model
# net()

def train():
    model = net()
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    train, val = preprocess_data()
    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
    model.fit(train, epochs=2, validation_data=val, callbacks=[tensorboard_callback])

    model.save(os.path.join('models', 'cellparasitized.h5'))

train()





