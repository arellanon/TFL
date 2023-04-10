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
#Import svm model
from sklearn.svm import SVC

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)

from libb import *
import logging

def getDataNumPy(filename):
    pathIn = "input/"
    X = np.load(pathIn + filename + ".npy")
    y = np.load(pathIn + filename + "_event.npy")
    print("X: ", X.shape)
    print("y: ", y.shape)
    return X, y

def getData(filename, filter_band = False, channels = None):
    pathIn = "input/fif/"
    epochs=mne.read_epochs(pathIn + filename + "_epo.fif", proj=True, preload=True, verbose=None)
    
    if filter_band: epochs.filter(8, 15)
    
    data = epochs.get_data(units='uV', picks=channels)
    
    event = epochs.events[:, -1]
    print("data: ", data.shape)
    print("event: ", event.shape)    
    
    logging.info('Filename: %s ', filename)
    logging.info('Filter Band (8-15 hz): %s ', filter_band)
    logging.info('Channel: %s ', channels)
    return data, event

def getResult(results, columns):
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    logging.info("Mean Accuracy: %.3f" % results.best_score_)
    logging.info("Config: %s" % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[
        columns
    ]
    return df


def getParamLDA():
    param_solver1 = {
        "LDA__solver": ["svd"],
        "LDA__store_covariance": [True, False],
        "LDA__tol": np.array([0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001])
    }
    
    param_solver2 = {
        "LDA__solver": ["lsqr", "eigen"],
        "LDA__shrinkage": [ None, "auto"] + np.arange(0, 1, 0.01).tolist()
    }
    
    param_grid = [param_solver1, param_solver2]
    return param_grid


def getParamCSP():
    param_solver1 = {
        "CSP__n_components": [2],
        "CSP__reg": [None],
        "CSP__log": [True, False],
        "CSP__norm_trace": [True, False]
    }
    
    param_grid = param_solver1
    return param_grid

def getParamSVM():
    param_solver1 = {
        "SVM__kernel": ["poly", "linear", "rbf", "sigmoid", "precomputed"]
    }
    
    param_grid = param_solver1
    return param_grid

def train(X, y, path, filename, param_grid, columns):
    logging.info("DATASET: %s" % filename)
    logging.info('Started')
    logging.info("X: %s ", X.shape)
    logging.info("y: %s ", y.shape)    
    
    #Modelo utiliza CSP y LDA
    
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)    
    svm = SVC(kernel='linear') # Linear Kernel

    #Modelo utiliza CSP y LDA
    model = Pipeline([('CSP', csp), ('SVM', svm)])
    
    
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1, verbose=-1)
    results = search.fit(X, y)
    
    df = getResult(results, columns)
    
    #Se guarda resultados y modelo
    df.to_csv(path + filename + ".csv", index=False)
    joblib.dump(search, path + filename + ".pkl")
    logging.info('Finished')
    
    
def run(output, param_grid, columns):
    logging.info('output: %s ', output)
    logging.info('param_grid: %s ', param_grid)
    logging.info('columns: %s ', columns)
    
    list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
        
    #DATASET: Dataset_1a
    X, y = getData("dataset_1", False)
    train(X, y, output, "Dataset_1a", param_grid, columns)
    #DATASET: Dataset_1b
    X, y = getData("dataset_1", True)
    train(X, y, output, "Dataset_1b", param_grid, columns)
    #DATASET: Dataset_1c
    X, y = getData("dataset_1", False, list_channel)
    train(X, y, output, "Dataset_1c", param_grid, columns)
    #DATASET: Dataset_1d
    X, y = getData("dataset_1", True, list_channel)
    train(X, y, output, "Dataset_1d", param_grid, columns)
    #DATASET: Dataset_2a
    X, y = getData("dataset_2", False)
    train(X, y, output, "Dataset_2a", param_grid, columns)
    #DATASET: Dataset_2b
    X, y = getData("dataset_2", True)
    train(X, y, output, "Dataset_2b", param_grid, columns)
    #DATASET: Dataset_3a
    X, y = getData("dataset_3", False)
    train(X, y, output, "Dataset_3a", param_grid, columns)
    #DATASET: Dataset_3b
    X, y = getData("dataset_3", True)
    train(X, y, output, "Dataset_3b", param_grid, columns)
    
    
def main():
    columns_lda=["param_LDA__solver", "param_LDA__shrinkage", "param_LDA__store_covariance", "param_LDA__tol", "params", "mean_test_score", "std_test_score"]
    columns_csp=["param_CSP__log", "param_CSP__n_components", "param_CSP__norm_trace", "param_CSP__reg", "params", "mean_test_score", "std_test_score"]
    
    columns_svm=["param_SVM__kernel", "params", "mean_test_score", "std_test_score"]
    
    logging.basicConfig(filename="example.log", filemode="w", level=logging.DEBUG, format='%(asctime)s %(message)s')
    
    """
    logging.info("LDA")
    output="output/lda/"
    param_grid = getParamLDA()
    columns = columns_lda
    
    run(output, param_grid, columns)
    
    logging.info("CSP")
    output="output/csp/"
    param_grid = getParamCSP()
    columns = columns_csp
    """

    logging.info("SVM")
    output="output/svm/"
    param_grid = getParamSVM()
    columns = columns_svm
    
    run(output, param_grid, columns)
    

if __name__ == "__main__":
    main()