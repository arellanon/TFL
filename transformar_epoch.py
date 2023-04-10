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

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *


def getData(filename, filter_band = False, channels = None):
    pathIn = "input/fif/"
    epochs=mne.read_epochs(pathIn + filename + "_epo.fif", proj=True, preload=True, verbose=None)
    
    if filter_band: epochs.filter(8, 15)
    
    data = epochs.get_data(units='uV', picks=channels)
    
    event = epochs.events[:, -1]
    print("data: ",data.shape)
    print("event: ",event.shape)
    
    return data, event



def main():
    list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    getData("dataset_2", False)

def main2():
    #epochs=mne.read_epochs("epochs/" + "BCICIV_calib_ds1d" + "-epo.fif", proj=True, preload=True, verbose=None)
    #epochs=mne.read_epochs("epochs/" + "Experiment6v4" + "-epo.fif", proj=True, preload=True, verbose='ERROR')
    epochs=mne.read_epochs("epochs/" + "Experiment7v1" + "-epo.fif", proj=True, preload=True, verbose=None)
    
    #list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    epochs.filter(8, 15)
    #data = epochs.get_data(units='uV', picks=list_channel)
    data = epochs.get_data(units='uV')
    event = epochs.events[:, -1]
    
    print("data: ", data.shape)
    print("event: ", event.shape)
    #Guardamos los set de datos
    filename = "Dataset_3b"
    path = "input/"
    
    np.save(path + filename, data)
    np.save(path + filename + "_event", event)
    
if __name__ == "__main__":
    main()