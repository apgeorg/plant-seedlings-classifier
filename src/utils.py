from __future__ import division
from __future__ import print_function
import cv2, os
import numpy as np
import pandas as pd
import datetime

def get_classes():
    return ['Black-grass',
            'Charlock',
            'Cleavers',
            'Common Chickweed',
            'Common wheat',
            'Fat Hen',
            'Loose Silky-bent',
            'Maize',
            'Scentless Mayweed',
            'Shepherds Purse',
            'Small-flowered Cranesbill',
            'Sugar beet']

def normalize(x):
    x = np.array(x, np.float32) / 255.
    #x = x.transpose((0, 1, 2, 3))
    print("Shape:", x.shape)
    return x

def get_image(path, img_width=48, img_height=48):
    img = cv2.imread(path)
    return cv2.resize(img, (img_width, img_height), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def load_train(path, categories, img_width, img_height):
    X_train, y_train, filenames = [], [], []
    for folder in categories:
        idx = categories.index(folder)
        print("ID {}: Load {}".format(idx, folder))
        fullpath = os.path.join(path, folder)
        for fl in os.listdir(fullpath):
            filename = os.path.basename(fl)
            img_path = os.path.join(fullpath, filename)
            img = get_image(img_path, img_width, img_height)
            X_train.append(img)
            filenames.append(filename)
            y_train.append(idx)
    return X_train, y_train, filenames

def load_test(path, img_width, img_height):
    X, filenames = [], []
    for fl in sorted(os.listdir(path)):
        filename = os.path.basename(fl)
        img_path = os.path.join(path, filename)
        img = get_image(img_path, img_width, img_height)
        X.append(img)
        filenames.append(filename)
    return X, filenames

def get_train(path, categories, img_width, img_height):
    X_train, y_train, filenames = load_train(path, categories, img_width, img_height)
    X_train = normalize(X_train)
    y_train = np.array(y_train, dtype=np.uint8)
    return X_train, y_train, filenames

def get_test(path, img_width, img_height):
    X_test, filenames = load_test(path, img_width, img_height)
    X_test = normalize(X_test)
    return X_test, filenames

def create_submission(preds, ids, output_path="./", filename="test", isSubmission=False):
    df = pd.DataFrame({"file": pd.Series(ids), "species": pd.Series(preds)})
    csvfile = filename
    if isSubmission:
        now = datetime.datetime.now()
        csvfile = "submission_" + filename + "_" + str(now.strftime("%Y-%m-%d-%H-%M"))
    df.to_csv(output_path + csvfile + ".csv", index=False)
    return df
    
def to_csv_ens(preds_true, preds, ids, classes, output_path="./", filename="test"):
    df1 = pd.DataFrame({"file": pd.Series(ids), "species": pd.Series(preds_true)})
    df2 = pd.DataFrame(preds, columns=classes)
    df = pd.concat([df1, df2], axis=1)
    df.to_csv(output_path + filename + ".csv", index=False)
    return df

