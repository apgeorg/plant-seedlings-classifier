from keras.layers import Flatten, Dense, Conv2D ,Dropout, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential
import utils

def create_model(input_shape=(128, 128, 3)):
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(Flatten())
    # Fully-Connection
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    # Output
    model.add(Dense(len(utils.get_classes()), activation='softmax'))

    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model