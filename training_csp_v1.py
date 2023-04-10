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


def getData():
    #epochs=mne.read_epochs("epochs/" + "BCICIV_calib_ds1d" + "-epo.fif", proj=True, preload=True, verbose=None)
    #epochs=mne.read_epochs("epochs/" + "Experiment7v1" + "-epo.fif", proj=True, preload=True, verbose=None)
    epochs=mne.read_epochs("epochs/" + "Experiment6v4" + "-epo.fif", proj=True, preload=True, verbose='ERROR')
    list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    epochs.filter(8, 15, list_channel)
    X = epochs.get_data(units='uV', picks=list_channel)
    y = epochs.events[:, -1]
    
    return X, y

def main(X, y):
    print("X: ", type(X), " - y: ", type(y))
    
    #csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)    
    csp = CSP()
    lda=LinearDiscriminantAnalysis()

    #Modelo utiliza CSP y LDA
    model = Pipeline([('CSP', csp), ('LDA', lda)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
    param_solver1 = {
        "CSP__n_components": [2],
        "CSP__reg": [None],
        "CSP__log": [True, False],
        "CSP__norm_trace": [True, False]
    }
    
    param_grid = param_solver1

    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=0)
    results = search.fit(X, y)
    
    
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[
        ["param_CSP__log", "param_CSP__n_components", "param_CSP__norm_trace", "param_CSP__reg", "params", "mean_test_score", "std_test_score"]
    ]
    
    df.to_csv('9-Experiment6v4_channel8_custom.csv', index=False)
    
    print(X.shape)
    joblib.dump(search, 'model_test.pkl')
    
if __name__ == "__main__":
    X, y = getData()
    print("mean: ", np.mean(X))
    print("median: ",np.median(X))
    print("std: ", np.std(X))
    print(y)
    main(X, y)