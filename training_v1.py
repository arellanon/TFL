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
    
    print(epochs.ch_names)
    #list_channel=['FP1', 'FP2', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2']
    #list_channel=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2']
    #epochs.filter(8, 15, ['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'])
    #epochs.filter(8, 15, list_channel)
    #epochs.filter(picks=list_channel)
    #epochs.filter(8, 15)
    
    #epochs.picks(['P3', 'P4', 'C3', 'C4', 'P7', 'P8', 'O1', 'O2'])
    
    #print(epochs)
    path = "model/"
    #Lo convierte a matriz numpy
    #X = epochs.get_data(picks=['C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 'O1', 'O2'], units='uV')
    #X = epochs.get_data(picks=list_channel, units='uV')
    X = epochs.get_data(units='uV')
    #print(epochs_data[0][0])
        
    #Se carga target (convierte 1 -> -1 y 2 -> 0 )
    #target = epochs.events[:, -1] - 2
    y = epochs.events[:, -1]
    return X, y

def main():
    X, y = getData()
    print("X: ", type(X), " - y: ", type(y))
    #print(epochs.events[:, -1])
    """
    print("print(target.shape):", target.shape)
    print("print(epochs_data.shape):", epochs_data.shape)
    #Se crea set de de pruebas y test
    X_train, X_test, y_train, y_test = train_test_split(epochs_data, target, test_size=0.2, random_state=0)
    
    print("print(X_train.shape):", X_train.shape)
    print("print(y_train.shape):", y_train.shape)
    
    #Guardamos los set de datos
    np.save(path + '/X_train.npy', X_train)
    np.save(path + '/y_train.npy', y_train)
    np.save(path + '/X_test.npy', X_test)
    np.save(path + '/y_test.npy', y_test)
    """  
    #Clasificadores del modelo
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)    
    ##SVD
    #lda=LinearDiscriminantAnalysis(solver='svd', n_components=1, shrinkage=None, priors=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
    lda=LinearDiscriminantAnalysis()

    #Modelo utiliza CSP y LDA
    model = Pipeline([('CSP', csp), ('LDA', lda)])
    #print("epochs_data: ", epochs_data.shape)
    #Entrenamiento del modelo
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    
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

    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
    results = search.fit(X, y)
    #print(sorted(results.cv_results_.keys()))
    
    print('Mean Accuracy: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    df = pd.DataFrame(results.cv_results_)[
        ["param_LDA__solver", "param_LDA__shrinkage", "param_LDA__store_covariance", "param_LDA__tol", "params", "mean_test_score", "std_test_score"]
    ]
    
    #df.to_csv('5-BCICIV_calib_ds1d_filter_channel8_default.csv', index=False)
    df.to_csv('9-Experiment6v4_channel8_custom.csv', index=False)
    #df.to_csv('8-Experiment7v1_channel8_custom.csv', index=False)
    
    print(X.shape)
    joblib.dump(search, 'model9.pkl')
    #display(df)
    """
    results = search.fit(X_train, y_train)
    #model.fit(X_train, y_train)
    
    score = model.score(X_train, y_train)
    print("Score entrenamiento: ", score)
    # plot CSP patterns estimated on full data for visualization
    #csp.fit_transform(epochs_data, target)
    #csp.plot_patterns(epochs.info, ch_type='eeg', size=1.5)
    
    #Resultados
    result=model.predict(X_test)
    
    #Guardamos el modelo
    joblib.dump(model, path + '/model.pkl')
    
    #Variables report
    ts = time.time()
    matriz=met.confusion_matrix(y_test, result)
    report=met.classification_report(y_test, result)
    
    #Mostrar report
    print(ts, ' - ', datetime.fromtimestamp(ts))
    print(matriz)
    print(report)
        
    #Archivo de salida
    fout=open(path + "/output.txt","a")
    fout.write(str(datetime.fromtimestamp(ts)) + "\n")
    fout.write(str(matriz) + "\n")
    fout.write(str( report))
    fout.write("\n")
    fout.close()
    """
    
if __name__ == "__main__":
    #X, y = getData()
    #print(X.shape)
    main()