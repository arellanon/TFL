#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 16:22:49 2021

@author: nahuel
"""
#librerias
import numpy as np
import time
from datetime import datetime
#from loaddata import *

#sklearn
from sklearn.model_selection import ShuffleSplit, cross_val_score, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import joblib

#mne
import mne
from mne.decoding import CSP
from mne.channels import read_layout
from mne.channels import make_standard_montage
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               compute_proj_ecg, compute_proj_eog)
from libb import *

def main():   
    #low_freq, high_freq = 7., 30.
    
    #raw = mne.io.read_raw_fif("data_eeg.fif", preload=True)
    #events_from_file = mne.read_events("event.fif",)
    path="DATA/"
    fileName = "data_eeg.fif"
    raw = mne.io.read_raw_fif(path + fileName, preload=True)
    #raw2 = mne.io.read_raw_fif("DATA/T13" + "/data_eeg.fif", preload=True)
    
    #raw.append(raw2)
    
    #sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample','sample_audvis_raw-eve.fif')
    events_from_file = mne.read_events(path + "data-eve.fif",)
    #print(type(raw))
    #print(type(events_from_file))
    #print(events_from_file)
    
    #raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')
    #Seleccionamos los canales a utilizar
    #raw.pick_channels(['P3', 'P4', 'C3', 'C4','P7', 'P8', 'O1', 'O2'])
    #print('raw select: ', raw.shape)
    
    #Seteamos la ubicacion de los canales segun el 
    #montage = make_standard_montage('standard_1020')
    #raw.set_montage(montage)
    
    #raw.plot(scalings='auto', n_channels=8, duration=20)
    #raw.plot(scalings='auto', n_channels=1, events=events)
    #raw.plot(scalings='auto', n_channels=1, events = events_from_file)
    #raw.plot(scalings='auto', n_channels=1)
    raw.plot(scalings=None, n_channels=8,  events = events_from_file)
    #raw.plot(scalings=None, n_channels=8)
    
if __name__ == "__main__":
    main()