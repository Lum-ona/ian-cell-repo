import os
import cv2
import keras
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

def load_n_preprocess_data():
    data = tf.keras.utils.image_dataset_from_directory('cell_images')
    scaled_data = data.map(lambda x,y: (x/255,y))

    train_size = int(len(scaled_data)*.7)+1
    val_size = int(len(scaled_data)* .2)
    test_size = int(len(scaled_data)* .1)

    train = scaled_data.take(train_size)
    val = scaled_data.skip(train_size).take(val_size)
    test = scaled_data.skip(train_size+val_size).take(test_size)
    print(len(train))

    return train, val

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

    model.summary()
    return model


def train():
    model = net()
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    logdir = 'logs'
    train, val = load_n_preprocess_data()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)
    
    model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    return model
# train().save(os.path.join('models', 'cellmodel-v1.h5'))


def test ():
    img = cv2.imread(os.path.join(DATA_DIR,'Uninfected', "C237ThinF_IMG_20151127_105345_cell_73.png"))
    resize = tf.image.resize(img, (256,256))  
    model = keras.models.load_model('models/cellmodel-v1.h5')
    yhat = model.predict(np.expand_dims(resize/255,0))

    #0 -> Parasitized
    #1 -> Uninfected

    if yhat < 0.5:
        print(f'Parasitized cell!')
    else:
        print(f'Uninfected cell!')

    print(yhat)
   
test()
    





