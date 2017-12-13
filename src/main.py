__author__ = 'apgeorg'

# Import libraries
import numpy as np
from keras import __version__
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import utils
import models

# *********************************
# Seed
seed = 42
np.random.seed(seed)
# Input and  dimensions
img_width, img_height = (64, 64)
input_shape = (img_width, img_height, 3)
# Modelname
modelname = 'myModel'
# *********************************

def train(X, y, epochs=1, batch_size=32):
    # y categorical
    y_true = np_utils.to_categorical(y, len(utils.get_classes()))

    # Split train/test data
    trX, teX, trY, teY = train_test_split(X, y_true, test_size=0.2, random_state=seed)

    # Create model
    model = models.create_model(input_shape)
    # Fit model
    model.fit(trX, trY, epochs=epochs, batch_size=batch_size)
    # Save model
    model.save('../models/' + modelname +'.h5')
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

    # Get test data
    X_test, X_test_id = utils.get_test('../input/test', img_width, img_height)
    # Predict on test data
    preds = model.predict(X_test, verbose=1)

    # Create submission
    utils.create_submission(lbenc.inverse_transform(preds), X_test_id, output_path="../submissions/", filename="simple_cnn", isSubmission=True)
    print('Finished.')

if __name__ == "__main__":
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("Competition: Plant Seedlings Classification")
    print("Author: ", __author__)
    print("Keras version: {}".format(__version__))
    print("+++++++++++++++++++++++++++++++++++++++++++")
    main()