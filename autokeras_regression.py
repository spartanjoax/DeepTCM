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
from keras.models import load_model
from keras.utils import plot_model

import autokeras as ak
from keras import backend as K
from tensorflow.keras.utils import CustomObjectScope

from helpers import compute_mean_std
from data import NASA_Dataset, StdScalerTransform, MinMaxScalerTransform

import json
import time

import logging
import datetime

sns.set_context("paper")
sns.set(font_scale = 1)
bgcolor=''
color=['hls','Plasma']

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

lists = {
    'list_model_name' : {},
    'list_model' : {},
    'list_model_file' : {},
    'list_r2_score' : {},
    'list_mae' : {},
    'list_rmse' : {},
    'list_mape' : {}
}


def root_mse(y_test, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_test)))  #mean_squared_error(y_test, y_pred,squared=False)

def get_scores(y_test, y_pred):
    rmse = float(tf.squeeze(root_mse(y_test, y_pred)).numpy())
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return rmse, r2, mae, mape

def evaluate_save_model(model, model_name, index, x_test, y_test, run_name, lists, folder=None):
    # Evaluate the model on the test set
    ypred = model.predict(x_test)
    rmse, r2, mae, mape = get_scores(y_test, ypred)
    print(f'RMSE: {rmse:.5f}\nR2: {r2:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}')
    
    file_name = f'{folder + "/" if folder is not None else ""}DL_{model_name}{run_name}_{rmse:.4f}.tf'

    lists['list_model'][index] = model
    lists['list_model_file'][index] = file_name
    lists['list_r2_score'][index] = r2
    lists['list_mae'][index] = mae
    lists['list_rmse'][index] = rmse
    lists['list_mape'][index] = mape

    model.save(file_name,save_format="tf")

    results_df = pd.DataFrame({'Ground truth' : np.squeeze(y_test), 'Prediction' : np.squeeze(ypred)},columns=['Ground truth', 'Prediction']).sort_values(by=['Ground truth'], ignore_index=True)

    plt.plot(results_df['Prediction'], label='Prediction')
    plt.plot(results_df['Ground truth'], label='Ground truth')
    plt.legend(loc='upper left')
    plt.savefig(f"{folder + '/' if folder is not None else ''}/DL_{model_name}{run_name}_test_pred.png")
    plt.clf()

def main(epochs=300, 
        sliding_window_size=250,
        sliding_window_stride=25,
        num_trials=10,
        run_name = '',
        preload_model_file = '',
        resume_training=False,
        block_type='CNN'):
    
    # Get the current date and time
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(f'AutoML_search_{timestamp}.log')
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')#logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    script_time = time.time()
    logger.info(f"---Starting script\n--- {(time.time() - script_time):.4f} seconds ---")    
  
    # Define the min and max for process parameters
    """
    NASA: 
    DOC - min: 0.75 - max: 1.5
    feed - min: 0.25 - max: 0.5
    material - min: 1 - max: 2
    Vc - min: 200.0 - max: 200.0

    MU-TCM: 
    DOC - 1.5
    feed - min: 0.05 - max: 0.2
    material - min: 1 - max: 2
    Vc - min: 50.0 - max: 200.0
    """
    proc_min = [0.1, 0.05, 1, 50] # DOC, feed, material, Vc
    proc_max = [1.5, 0.5, 2, 200] # DOC, feed, material, Vc
    minmax_transform = MinMaxScalerTransform(min=proc_min, max=proc_max)
    
    run = run_name[1:]
    split_ratios = (0.48, 0.12, 0.4) 
    # Compute mean and std for the training data
    train_dataset = NASA_Dataset(split='train', signal_group=run, split_ratios=split_ratios)
    mean, std = compute_mean_std(train_dataset)
    # Define normalization transform
    normalize_transform = StdScalerTransform(mean=mean, std=std)

    nasa_train_base = NASA_Dataset(split='train', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride)
    nasa_val_base = NASA_Dataset(split='val', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride)
    nasa_test_base = NASA_Dataset(split='test', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride)

    xtr, xtr_proc, ytr = nasa_train_base.data.cpu().data.numpy(), nasa_train_base.proc_data.cpu().data.numpy(), nasa_train_base.targets.cpu().data.numpy()
    xval, xval_proc, yval = nasa_val_base.data.cpu().data.numpy(), nasa_val_base.proc_data.cpu().data.numpy(), nasa_val_base.targets.cpu().data.numpy()
    xte, xte_proc, yte = nasa_test_base.data.cpu().data.numpy(), nasa_test_base.proc_data.cpu().data.numpy(), nasa_test_base.targets.cpu().data.numpy()

    logger.info(f'X train: {xtr.shape}')
    logger.info(f'X valid: {xval.shape}')
    logger.info(f'X test: {xte.shape}')
    logger.info(f'Proc train: {xtr_proc.shape}')
    logger.info(f'Proc valid: {xval_proc.shape}')
    logger.info(f'Proc test: {xte_proc.shape}')
    logger.info(f'Y train: {ytr.shape}')
    logger.info(f'Y valid: {yval.shape}')
    logger.info(f'Y test: {yte.shape}')

    folder = 'auto_ml'
    model_name = "AutoKeras_"+block_type.upper()
    index = 0
    start_time = time.time()
    lists['list_model_name'][index] = model_name

    print(f'Look for file {folder}/DL_{model_name}_{run}_*.tf')
    matches = glob.glob(f'{folder}/DL_{model_name}_{run}_*.tf')
    if len(matches) > 0 and resume_training:
        logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
        best_model = load_model(matches[0], compile=False)
        logger.info(f'================== Model {model_name} for run {run} loaded')
    else:
        logger.info(f'================== Training {model_name} for run {run}')
        if block_type.lower() == "cnn":
            signal_input = ak.ImageInput()
            signal_output = ak.ImageBlock(normalize=False,augment=False)(signal_input)
        elif "rnn" in block_type.lower():
            signal_input = ak.Input()
            rnn_block = ak.RNNBlock(return_sequences=True, layer_type=('gru' if 'gru' in block_type.lower() else 'lstm'))(signal_input)
            signal_output = ak.RNNBlock(return_sequences=True, layer_type=('gru' if 'gru' in block_type.lower() else 'lstm'))(rnn_block)
        structured_input = ak.Input()
        structured_output = ak.DenseBlock(num_layers=2, num_units=32, use_batchnorm=True, dropout=0.25)(structured_input)
        merge = ak.Merge()([signal_output, structured_output])
        output = ak.DenseBlock(num_layers=2, num_units=32, use_batchnorm=True, dropout=0.25)(merge)
        regression_output = ak.RegressionHead(metrics=[root_mse], dropout=0)(output)
        
        clf = ak.AutoModel(
            inputs=[signal_input, structured_input],
            outputs=regression_output, directory=folder,
            overwrite=not resume_training, max_trials=num_trials, project_name=f'DL_{model_name}{run_name}',
            tuner='bayesian') #objective = keras_tuner.Objective("val_root_mse", direction="min"), 

        # Feed the image classifier with training data.
        history = clf.fit([xtr, xtr_proc], ytr, epochs=epochs, validation_data=([xval, xval_proc], yval))#, callbacks=[early_stop])
        logger.info(f'================== {model_name} Model history')
        logger.info(history.history.keys())
        dump(history, f'{folder}/DL_{model_name}{run_name}_history.bin')
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.legend()
        plt.savefig(f"{folder}/DL_{model_name}_training_loss{run_name}.png")
        plt.clf()

        best_hp = clf.tuner.get_best_hyperparameters(num_trials=3)[0]
        logger.info(best_hp.values)
        with open(f"{folder}/DL_{model_name}_best_hp{run_name}.json", "w") as f:
            # Dump the best_hp.values dictionary to the file
            json.dump(best_hp.values, f, indent=4)
    
        with CustomObjectScope({'root_mse': root_mse}):
            best_model = clf.export_model()
            
            for index, model in enumerate(clf.tuner.get_best_models(num_models=3)):
                plot_model(model, to_file=f'{folder}/DL_{model_name}{run_name}_modelplot_{index}.png', show_shapes=True, show_layer_names=True)
                file_name = f'{folder}/DL_{model_name}{run_name}_{index}.tf'
                model.save(file_name,save_format="tf")
        
    best_model.summary()
    plot_model(best_model, to_file=f'{folder}/DL_{model_name}{run_name}_modelplot.png', show_shapes=True, show_layer_names=True)
    evaluate_save_model(best_model, model_name, index, [xte, xte_proc], yte, run_name, lists, folder=folder)
    logger.info("--- %s seconds ---" % (time.time() - start_time))
    logger.info(f'================== {model_name} model trained')

    lists_dict = {
                    'ModelName':lists['list_model_name'],
                    'RMSE':lists['list_rmse'],
                    'R2':lists['list_r2_score'],
                    'MAE':lists['list_mae']
                }
    df = pd.DataFrame(lists_dict)
    df.to_csv(f'{folder}/DL_{model_name}{run_name}_scores.csv',sep=';',decimal='.')
    logger.info(df)

    logger.info('================== Training finished')

    #plot bw RMSE of models
    sns.barplot(x=df['ModelName'],y=df['RMSE']).set(title="RMSE of models (closer to 0 is better)")
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees 
    plt.savefig(f"{folder}/DL_{model_name}{run_name}_RMSE_scores.png")
    plt.clf()
    #plot bw R2 of models
    sns.barplot(x=df['ModelName'],y=df['R2']).set(title="R2 score of models (closer to 1 is better)")
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks by 45-degrees
    plt.tight_layout()
    plt.savefig(f"{folder}/DL_{model_name}{run_name}_R2_scores.png")
    plt.clf()
        
    logger.info(f"---Script time\n--- {(time.time() - script_time):.4f} seconds ---")

if __name__ == "__main__":
    
    epochs=300
    sliding_window_size=500
    sliding_window_stride=50
    preload_model_file=''
    resume_training=True

    print('Args received: ', sys.argv[1:])

    opts, args = getopt.getopt(sys.argv[1:],"he:p:c:r:")
    for opt, arg in opts:
      if opt == '-h':
        print ('DL_ensemble_regression_mm.py\n\t-a <apply_averaging (1 or 0)>\n\t-e <epochs>\n\t-w <average_window_size>\n\t-p <preload_model_file>\n\t-r <resume_training (1 or 0)>')
        sys.exit()
      elif opt in ("-e"):
        epochs = int(arg)
      elif opt in ("-p"):
        preload_model_file = arg
      elif opt in ("-r"):
        resume_training = arg == "1"

    if resume_training:
        print(f'================== Resuming training')
    else:
        print(f'================== Starting new training')
        
    print(f'================== Training for {epochs} epochs')
    print(f'================== Augmenting data with a sliding window size of {sliding_window_size} and a stride of {sliding_window_stride}')
    print(f'================== Preload model file: {preload_model_file}')

    print('Press any key to begin')
    inp = input()

    for run_name in ['_all']:#['_AC','_AC_AE','_AC_Vib','_AC_table','_ACDC']:
        num_trials = 50# if run_name == '_AC' else 40
        for block_type in ['rnn_gru','rnn','cnn']:
            print(f'================== Training {run_name} - {block_type}')
            main(epochs=epochs, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                num_trials=num_trials, run_name=run_name,
                preload_model_file=preload_model_file, resume_training=resume_training, block_type=block_type.upper())