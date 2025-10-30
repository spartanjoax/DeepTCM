#https://autokeras.com/
#https://autokeras.com/tutorial/faq/#how-to-resume-a-previously-killed-run

#https://pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/

import sys, getopt
import glob

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump
from pickle import load


from keras.models import Model, load_model
from keras.utils import plot_model
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Bidirectional, GRU, concatenate, Dropout, BatchNormalization, Activation, Add, Multiply
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

from helpers import compute_mean_std
from data import NASA_Dataset, StdScalerTransform, MinMaxScalerTransform

import json
import time

import logging
import datetime

def main(run = ''):

    results = []
    model_list = []

    model_list = ['LSTM','BiGRU','CNN','Ensemble','ResNet','RobustResNet','ResNetProc','RobustResNetProc']

    folder = 'manual'
    if os.path.exists(f'{folder}/DL_{run}_scores.csv'):

        for model_name in model_list:
            print('\n================== Plotting', model_name)
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_history.bin')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_history.bin')
            if len(matches) > 0:
                with open(f'{folder}/DL_{model_name}_{run}_history.bin', 'rb') as pickle_file:
                    history = load(pickle_file)
                
                plt.plot(history.history['loss'], label='Plotting')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_training_loss.png")
                plt.clf()

if __name__ == "__main__":
    
    for run_name in ['AC','AC_table','DC','DC_table','ACDC','all']:
        main(run=run_name)