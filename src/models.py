from keras.layers import Flatten, Dense, Conv2D ,Dropout, MaxPooling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import utils

def get_model2(input_shape=(128, 128, 3)):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten
    model.add(Flatten())
    # Fully-Connection
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    # Output
    model.add(Dense(len(utils.get_classes()), activation='softmax'))

    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_conv_model(model):
    last_conv_idx = [i for i, layer in enumerate(model.layers) if type(layer) is Conv2D][-1]
    layers = model.layers[:last_conv_idx+1]
    return Model(inputs=model.input, outputs=layers[-1].output)

def stack_on_top(p, model):
    inp = model.output
    x = MaxPooling2D()(inp)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(p / 4)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(p)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(p / 2)(x)
    y = Dense(len(utils.get_classes()), activation='softmax')(x)
    model = Model(inputs=[model.input], outputs=[y])
    return model

def get_VGG16(input_shape=(128, 128, 3)):
    model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    model = create_conv_model(model)
    model = stack_on_top(0.6, model)
    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_InceptionV3(input_shape=(299, 299, 3)):
    model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    model = create_conv_model(model)
    model = stack_on_top(0.6, model)
    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_ResNet50(input_shape=(224, 224, 3)):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    model = create_conv_model(model)
    model = stack_on_top(0.6, model)
    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_InceptionResNetV2(input_shape=(299, 299, 3)):
    model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    model = create_conv_model(model)
    model = stack_on_top(0.6, model)
    optimizer = Adam(1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
