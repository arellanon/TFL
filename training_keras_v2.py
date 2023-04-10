#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 23:59:47 2020

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
#from loaddata import *
import pandas as pd

#sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import joblib

#from keras.preprocessing.image import load_img
from keras.utils import load_img, img_to_array
#from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *


def getDataBK():
    # load an image from file
    image = load_img('mug.jpg', target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    print(image.shape)
    print(type(image))
    return image


def getData():
    #epochs=mne.read_epochs("epochs/" + "BCICIV_calib_ds1d" + "-epo.fif", proj=True, preload=True, verbose=None)
    #epochs=mne.read_epochs("epochs/" + "Experiment7v1" + "-epo.fif", proj=True, preload=True, verbose=None)
    epochs=mne.read_epochs("epochs/" + "Experiment6v4" + "-epo.fif", proj=True, preload=True, verbose='ERROR')
    list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    epochs.filter(8, 15, list_channel)
    X = epochs.get_data(units='uV', picks=list_channel)
    y = epochs.events[:, -1]
        
    return X

def main():
    data = getData()
    # load the model
    model = VGG16()
    data = preprocess_input(data)
    # predict the probability across all output classes
    yhat = model.predict(data)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    # retrieve the most likely result, e.g. highest probability
    label = label[0][0]
    # print the classification
    print('%s (%.2f%%)' % (label[1], label[2]*100))
    
if __name__ == "__main__":
    main()