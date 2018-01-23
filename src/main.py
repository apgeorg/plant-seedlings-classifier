__author__ = 'apgeorg'

# Import libraries
import numpy as np
from keras import __version__
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model
import utils, models

# *********************************
# Seed
SEED = 42
np.random.seed(SEED)
# Input and  dimensions
img_width, img_height = (299, 299)
input_shape = (img_width, img_height, 3)
# Modelname
modelname = 'inceptV3-4'
# *********************************

def get_callbacks(path):
    early_stop = EarlyStopping('val_loss', patience=5, mode="min")
    model_ckpt = ModelCheckpoint(path, save_best_only=True)
    return [early_stop, model_ckpt]

def image_augmetation(X, y, batch_size=32):
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,
                                 width_shift_range=0.1, height_shift_range=0.1,
                                 zoom_range=0.1, rotation_range=90)
    datagen.fit(X)
    return datagen.flow(X, y, batch_size=batch_size, seed=SEED)

def train(X, y, epochs=1, batch_size=32):
    # y categorical
    y_true = np_utils.to_categorical(y, len(utils.get_classes()))

    # Split train/test data
    trX, teX, trY, teY = train_test_split(X, y_true, test_size=0.2, random_state=SEED)

    # Image augmentation
    gen = image_augmetation(trX, trY)

    # Create model
    #model = models.get_model2(input_shape)
    model = models.get_InceptionV3(input_shape)
    #model = models.get_ResNet50(input_shape)
    #model = models.get_InceptionResNetV2(input_shape)

    # Fit model
    #model.fit(trX, trY, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

    # Fit model (generator)
    try:
        model.fit_generator(gen, epochs=epochs,
              steps_per_epoch=len(X)/batch_size,
              validation_data=(teX, teY),
              callbacks=get_callbacks(path='../models/' + modelname + '-{epoch:02d}-{val_loss:.3f}' +'.h5')
              )
        # Save model
        model.save('../models/' + modelname + '.h5')
    except:
        # Save model on keyboard abort
        model.save('../models/' + modelname + '_OnExit' + '.h5')

    print("Model saved.")
    return model

# Main function
def main():
    # Get label encoder
    lb = LabelBinarizer()
    lbenc = lb.fit(utils.get_classes())

    # Get train data
    X_train, y_train, train_filenames = utils.get_train('../input/train', list(lbenc.classes_), img_width, img_height)

    # Create and train model
    model = train(X_train, y_train, epochs=100, batch_size=32)

    print("+++++++++++++++++++++++++++++++++++++++++++")

    # Load model ...
    #model = load_model('../models/'+ 'model2_f0.86/'+ 'model2-64-0.341.h5')

    # Get test data
    X_test, X_test_id = utils.get_test('../input/test', img_width, img_height)
    # Predict on test data
    preds = model.predict(X_test, verbose=1)

    # Create submission
    utils.create_submission(lbenc.inverse_transform(preds), X_test_id, output_path="../submissions/", filename=modelname, isSubmission=True)
    utils.to_csv_ens(lbenc.inverse_transform(preds), preds, X_test_id, utils.get_classes(), output_path="../submissions/", filename=modelname):
    print('Finished.')

if __name__ == "__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Competition: Plant Seedlings Classification")
    print("Author: ", __author__)
    print("Keras version: {}".format(__version__))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    main()
