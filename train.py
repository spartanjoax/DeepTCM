#https://autokeras.com/
#https://autokeras.com/tutorial/faq/#how-to-resume-a-previously-killed-run

#https://pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/

import sys, getopt
import glob
import gc

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["KERAS_BACKEND"] = "torch" #Available backend options are: "jax", "tensorflow", "torch"
#import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump
from pathlib import Path

from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import saving

from models import create_xception, create_resnet, create_lstm, create_bigru, create_cnn, create_ensemble, create_transformer, root_mse

# Clear all previously registered custom objects
saving.get_custom_objects().clear()

from helpers import compute_mean_std, compute_min_max, convert_to_cwt, convert_to_stransform, convert_to_fsst, convert_to_wsst, get_scores#, TrainableHighPass, DownSample2
from data import NASA_Dataset, StdScalerTransform, MinMaxScalerTransform
from models.generative import train_generative_model, ConditionalVAE, TimeGAN, TimeGrad
import os

from keras import backend as K

import json
import time

import logging
import datetime

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

sns.set_context("paper")
sns.set(font_scale = 1)
bgcolor=''
color=['hls','Plasma']

use_big_model = False
es_patience=50
es_patience_lstm=50
es_min_delta=0.0005
split_type='runs'
split_ratios = (0.64, 0.16, 0.2) #(0.48, 0.12, 0.4) 
generative_model_name = 'ConditionalVAE'
gen_suffix = 'cvae'

data_transform_cwt = [
    'cwt',
    'fsst',
    'wsst',
    None
]

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def get_scores(y_test, y_pred):
    rmse = float(np.squeeze(root_mse(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return rmse, r2, mae, mape

def evaluate_save_model(model, model_name, index, x_test, y_test, 
                        run_name, results, model_results, folder=None, 
                        y_min=0, y_max=1, norm_target=False):
    # Evaluate the model on the test set
    start_time = time.time()
    ypred = model.predict(x_test)
    total_time = time.time() - start_time
    model_results['eval_time'] = total_time
    
    # Inverse MinMax predictions and targets
    if norm_target:
        ypred_orig = ypred * (y_max - y_min) + y_min
        y_test_orig = y_test * (y_max - y_min) + y_min
    else:
        ypred_orig = ypred
        y_test_orig = y_test
    
    rmse, r2, mae, mape = get_scores(y_test_orig, ypred_orig)
    print(f'RMSE: {rmse:.5f}\nR2: {r2:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}')
    
    file_name = f'{folder + "/" if folder is not None else ""}DL_{model_name}_{run_name}_{rmse:.4f}.h5'

    model_results['model_file'] = file_name
    model_results['r2_score'] = r2
    model_results['rmse'] = rmse
    results.append(model_results)

    model.save(file_name)

    results_df = pd.DataFrame({'Ground truth' : np.squeeze(y_test_orig), 'Prediction' : np.squeeze(ypred_orig)},columns=['Ground truth', 'Prediction']).sort_values(by=['Ground truth'], ignore_index=True)

    plt.plot(results_df['Prediction'], label='Prediction')
    plt.plot(results_df['Ground truth'], label='Ground truth')
    plt.legend(loc='upper left')
    plt.savefig(f"{folder + '/' if folder is not None else ''}/DL_{model_name}_{run_name}_test_pred.png")
    plt.clf()

def train_model(model_name, create_fn, train_data, val_data, test_data, 
                folder, run, cwt_transform, epochs, resume_training, 
                logger, index, results, cwt_times, model_params=None, fit_params=None, 
                patience=50, plot_graph=False, y_min=0, y_max=1, norm_target=False):
    
    xtr, ytr = train_data
    xval, yval = val_data
    xte, yte = test_data
    
    cwt_time_tr, cwt_time_te = cwt_times

    logger.info(f'\n================== Training {model_name} with {cwt_transform} as transform')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, min_delta=es_min_delta, restore_best_weights=True)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    clr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_delta=0.001, min_lr=1e-5)
    
    start_time = time.time()
    model_results = {
        'model_name' : model_name,
        'scalogram' : str(cwt_transform),
        'cwt_time_tr': cwt_time_tr,
        'cwt_time_te': cwt_time_te,
    }

    print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
    matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
    
    model = None
    if len(matches) > 0 and resume_training:
        logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
        model = load_model(matches[0], compile=False)
        with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
            model_results = json.load(fp)
        total_time = model_results['time']
        logger.info(f'================== Model {model_name} for run {run} loaded')
    else:
        logger.info(f'================== Training {model_name} for run {run}')
        
        if model_params is None:
            model_params = {}
        model = create_fn(**model_params)
        
        if plot_graph:
             plot_model(model, show_shapes=True, to_file=f'{folder}/DL_{model_name}_graph_{run}_{cwt_transform}.png')

        # Fit the model
        if fit_params is None:
            fit_params = {}
            
        history = model.fit(xtr, ytr, batch_size=16, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc, clr], **fit_params)
        
        logger.info(f'================== {model_name} Model history')
        logger.info(history.history.keys())
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.legend()
        plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
        plt.clf()
        
        model = load_model('best_model.h5', compile=False)       
        total_time = time.time() - start_time
        model_results['time'] = total_time
        model_results['history'] = history.history
        
    evaluate_save_model(model, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder, y_min=y_min, y_max=y_max, norm_target=norm_target)
    with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
        outfile.write(json.dumps(model_results, indent=4, cls=NumpyFloatValuesEncoder))
        
    logger.info("--- %s seconds ---" % (total_time))
    logger.info(f'================== {model_name} model trained')
    
    return model
    
def create_proc_model(base_model_fn, base_model_params, ensemble_params):
    base_model = base_model_fn(**base_model_params)
    ensemble_params['model_list'] = [base_model]
    return create_ensemble(**ensemble_params)

def get_expected_models(test_mode=False, use_big_model=False):
    models = []
    # Basic models
    if use_big_model:
        models.append('Xception')
    models.append('CNN')
    
    if not test_mode:
        models.extend(['LSTM', 'BiGRU', 'Transformer'])
        # Ensembles
        models.extend(['Ensemble', 'EnsembleProc'])
        # ResNets
        models.extend(['ResNet', 'RobustResNet'])
        # Proc models (CNN is always in, others if not test_mode)
        models.extend(['CNNProc', 'LSTMProc', 'BiGRUProc', 'ResNetProc', 'RobustResNetProc', 'TransformerProc'])
    else:
        # In test mode, only CNN based proc
        models.append('CNNProc')
        
    return models

def check_run_complete(folder, run, transforms, test_mode=False, use_big_model=False):
    expected_models = get_expected_models(test_mode, use_big_model)
    
    # If folder doesn't exist, runs are definitely not complete
    if not os.path.exists(folder):
        return False
        
    for transform in transforms:
        for model in expected_models:
            # Check for any file matching the pattern DL_{model}_{run}_{transform}_*.h5
            pattern = f'{folder}/DL_{model}_{run}_{transform}_*.h5'
            matches = glob.glob(pattern)
            if len(matches) == 0:
                print(f"Missing model: {pattern}")
                return False
                
    return True


def main(epochs=300, 
        sliding_window_size=250,
        sliding_window_stride=25,
        run = '',
        resume_training=False,
        seed=42,
        test_mode=False,
        norm_target=False,
        runs_per_case_test_val=1,
        split_offset=0,
        use_generative=False,
        extract_features='both'):
    results = []
    
    folder = f'manual_sw{sliding_window_size}_ss{sliding_window_stride}_seed{seed}'
    if norm_target:
        folder += '_targetnorm'
    if runs_per_case_test_val > 0:
        folder += f'_rpc{runs_per_case_test_val}_off{split_offset}'
    if use_generative:
        folder += '_'+gen_suffix
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    # Generative Model Handling
    if use_generative:
        output_dir = 'generative_results'
        suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
        gen_model_file = os.path.join(output_dir, f"{generative_model_name}_{suffix}.weights.h5")
        gen_scaler_file = os.path.join(output_dir, f"{generative_model_name}_scaler_{suffix}.joblib")
        
        if not os.path.exists(gen_model_file) or not os.path.exists(gen_scaler_file):
            print(f"================== {generative_model_name} Model or Scaler not found. Training...")
            model_map = {
                'ConditionalVAE': ConditionalVAE,
                'TimeGAN': TimeGAN,
                'TimeGrad': TimeGrad
            }
            if generative_model_name not in model_map:
                raise ValueError(f"Unknown generative model name: {generative_model_name}")
            
            model_class = model_map[generative_model_name]
            
            # Use a temporary dataset to train the generative model
            # We use 'all' signal group for a general model
            training_dataset = NASA_Dataset(split='train', signal_group='all', 
                                         sliding_window_size=sliding_window_size, 
                                         sliding_window_stride=sliding_window_stride)
            
            train_generative_model(model_class, training_dataset, sliding_window_size, sliding_window_stride, 
                                   latent_dim=16, epochs=200, output_dir=output_dir)
        else:
            print(f"================== {generative_model_name} Model found. Using existing weights.")

    # Check if run is complete
    if check_run_complete(folder, run, data_transform_cwt, test_mode=test_mode, use_big_model=use_big_model) and resume_training:
        print(f'================== Run {run} complete. Skipping...')
        return

    # Get the current d.ate and time
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(f'{folder}/manual_{timestamp}.log')
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
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
    #proc_min = [0.1, 0.05, 1, 50] # DOC, feed, material, Vc
    #proc_max = [1.5, 0.5, 2, 200] # DOC, feed, material, Vc

    # Compute mean and std for the training data
    train_dataset = NASA_Dataset(split='train', signal_group=run, split_ratios=split_ratios,sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset,
                                    use_generative=use_generative, generative_model_name=generative_model_name, extract_features=extract_features!="no")
    mean, std = compute_mean_std(train_dataset)
    
    # Compute min and max for process params (now includes time-domain features)
    proc_min, proc_max = compute_min_max(train_dataset)
    print(f"Process parameter min: {proc_min} max: {proc_max}")
    proc_min[0] = 0.75
    proc_min[1] = 0.05
    proc_min[2] = 1
    proc_min[3] = 50
    proc_max[0] = 1.5
    proc_max[1] = 0.5
    proc_max[2] = 2
    proc_max[3] = 200
    print(f"Process parameter min: {proc_min} max: {proc_max}")
    minmax_transform = MinMaxScalerTransform(min=proc_min, max=proc_max)
    
    # Define normalisation transform
    normalize_transform = StdScalerTransform(mean=mean, std=std)

    nasa_train_base = NASA_Dataset(split='train', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset,
                                    use_generative=use_generative, generative_model_name=generative_model_name, extract_features=extract_features!="no")
    logger.info(f"Train runs: {nasa_train_base.train_runs}")
    nasa_val_base = NASA_Dataset(split='val', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset,
                                    extract_features=extract_features!="no")
    logger.info(f"Val runs: {nasa_train_base.val_runs}", )
    nasa_test_base = NASA_Dataset(split='test', signal_group=run, split_ratios=split_ratios, transformX=normalize_transform, transformProc=minmax_transform, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset,
                                    extract_features=extract_features!="no")
    logger.info(f"Test runs: {nasa_train_base.test_runs}", )

    xtr, xtr_proc, ytr = nasa_train_base.data.cpu().data.numpy(), nasa_train_base.proc_data.cpu().data.numpy(), nasa_train_base.targets.cpu().data.numpy()
    xval, xval_proc, yval = nasa_val_base.data.cpu().data.numpy(), nasa_val_base.proc_data.cpu().data.numpy(), nasa_val_base.targets.cpu().data.numpy()
    xte, xte_proc, yte = nasa_test_base.data.cpu().data.numpy(), nasa_test_base.proc_data.cpu().data.numpy(), nasa_test_base.targets.cpu().data.numpy()
    
    # Normalize targets
    if norm_target:
        y_min = 0
        y_max = 0.4
    
        ytr = (ytr - y_min) / (y_max - y_min)
        yval = (yval - y_min) / (y_max - y_min)
        yte = (yte - y_min) / (y_max - y_min)
    else:
        y_min = 0
        y_max = 1
    
    logger.info(f'X train: {xtr.shape}')
    logger.info(f'X valid: {xval.shape}')
    logger.info(f'X test: {xte.shape}')
    logger.info(f'Proc train: {xtr_proc.shape}')
    logger.info(f'Proc valid: {xval_proc.shape}')
    logger.info(f'Proc test: {xte_proc.shape}')
    logger.info(f'Y train: {ytr.shape}')
    logger.info(f'Y valid: {yval.shape}')
    logger.info(f'Y test: {yte.shape}')
    
    index = 0      

    #plot_signals(xtr, nasa_train_base.get_signal_list(), signal_chanel=2)
        
    for cwt_transform in data_transform_cwt:
        if cwt_transform is not None:
            logger.info(f'================== Generating {cwt_transform} CWT scalograms...')
            cwt_time_tr = time.time()
            if cwt_transform == 'strans':
                xtr_cwt = convert_to_stransform(xtr, gamma=0.1, description='S-transform for train set')
                xval_cwt = convert_to_stransform(xval, gamma=0.1, description='S-transform for validation set')
            elif cwt_transform == 'fsst':
                xtr_cwt = convert_to_fsst(xtr, fs=250, description='FSST for train set')
                xval_cwt = convert_to_fsst(xval, fs=250, description='FSST for validation set')
            elif cwt_transform == 'wsst':
                xtr_cwt = convert_to_wsst(xtr, fs=250, description='WSST for train set')
                xval_cwt = convert_to_wsst(xval, fs=250, description='WSST for validation set')
            else:
                xtr_cwt = convert_to_cwt(xtr, description='CWT for train set')
                xval_cwt = convert_to_cwt(xval, description='CWT for validation set')
            
            logger.info('Normalizing CWT scalograms...')

            cwt_mean = np.mean(xtr_cwt, axis=(0, 1, 2), keepdims=True)  # (1, 1, 1, C)
            cwt_scale = np.std(xtr_cwt, axis=(0, 1, 2), keepdims=True) + 1e-8

            logger.info(f'X CWT mean: {cwt_mean.shape}')
            logger.info(f'X CWT scale: {cwt_scale.shape}')

            xtr_cwt = (xtr_cwt - cwt_mean) / cwt_scale
            xval_cwt = (xval_cwt - cwt_mean) / cwt_scale
            
            cwt_time_tr = time.time() - cwt_time_tr

            logger.info(f'X CWT for train and val calculated in {cwt_time_tr:.2f} seconds')
            
            cwt_time_te = time.time()
            if cwt_transform == 'strans':
                xte_cwt = convert_to_stransform(xte, description='S-transform for test set')
            elif cwt_transform == 'fsst':
                xte_cwt = convert_to_fsst(xte, fs=250, description='FSST for test set')
            elif cwt_transform == 'wsst':
                xte_cwt = convert_to_wsst(xte, fs=250, description='WSST for test set')
            else:
                xte_cwt = convert_to_cwt(xte, description='CWT for test set')
            
            xte_cwt = (xte_cwt - cwt_mean) / cwt_scale
            cwt_time_te = time.time() - cwt_time_te
            
            logger.info(f'X CWT train: {xtr_cwt.shape}')
            logger.info(f'X CWT valid: {xval_cwt.shape}')
            logger.info(f'X CWT test: {xte_cwt.shape}')
            logger.info(f'================== {cwt_transform} CWT scalograms generated...')
        else:
            cwt_time_tr = 0
            cwt_time_te = 0
            xtr_cwt = np.copy(xtr)
            xval_cwt = np.copy(xval)
            xte_cwt = np.copy(xte)

        trained_models = {}

        model_configs = [
            {
                'name': 'Xception',
                'create_fn': create_xception,
                'params': {'learning_r': 1e-3, 'decay': 1e-5, 'x_shape': xtr_cwt.shape, 'scalogram_cwt': cwt_transform},
                'data': (xtr_cwt, xval_cwt, xte_cwt),
                'patience': es_patience
            },
            {
                'name': 'CNN',
                'create_fn': create_cnn,
                'params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'scalogram_cwt': cwt_transform},
                'data': (xtr_cwt, xval_cwt, xte_cwt),
                'patience': es_patience
            },
            {
                'name': 'LSTM',
                'create_fn': create_lstm,
                'params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr.shape},
                'data': (xtr, xval, xte), # LSTM uses raw data
                'patience': es_patience_lstm,
                'epochs': 500
            },
            {
                'name': 'BiGRU',
                'create_fn': create_bigru,
                'params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr.shape},
                'data': (xtr, xval, xte), # BiGRU uses raw data
                'patience': es_patience_lstm,
                'epochs': 500
            },
            {
                'name': 'Transformer',
                'create_fn': create_transformer,
                'params': {'learning_r': 1e-4, 'decay': 1e-5, 'dropout': 0.1, 'x_shape': xtr.shape, 'num_heads': 4, 'embed_dim': 64, 'num_blocks': 4, 'scalogram_cwt': None},
                'data': (xtr, xval, xte), # 1D Transformer uses raw data
                'patience': es_patience,
                'epochs': 500
            }
        ]
        
        if not use_big_model:
            model_configs = model_configs[1:]

        # Filter based on extract_features
        if extract_features == 'yes':
            model_configs = [config for config in model_configs if config['name'] in ['CNN', 'LSTM', 'BiGRU']]

        if test_mode:
            model_configs = [config for config in model_configs if config['name'] == 'CNN']

        for config in model_configs:
            index += 1
            model = train_model(
                model_name=config['name'],
                create_fn=config['create_fn'],
                train_data=(config['data'][0], ytr),
                val_data=(config['data'][1], yval),
                test_data=(config['data'][2], yte),
                folder=folder,
                run=run,
                cwt_transform=cwt_transform,
                epochs=config.get('epochs', epochs),
                resume_training=resume_training,
                logger=logger,
                index=index,
                results=results,
                cwt_times=(cwt_time_tr, cwt_time_te),
                model_params=config['params'],
                patience=config['patience'],
                y_min=y_min,
                y_max=y_max,
                norm_target=norm_target
            )
            trained_models[config['name']] = model

        if all(k in trained_models for k in ['LSTM', 'BiGRU', 'CNN']) and not test_mode:
            if extract_features != 'yes':
                index += 1
                ensemble_input = [xtr, xtr, xtr_cwt]
                ensemble_val = [xval, xval, xval_cwt]
                ensemble_test = [xte, xte, xte_cwt]
                
                train_model(
                    model_name='Ensemble',
                    create_fn=create_ensemble,
                    train_data=(ensemble_input, ytr),
                    val_data=(ensemble_val, yval),
                    test_data=(ensemble_test, yte),
                    folder=folder,
                    run=run,
                    cwt_transform=cwt_transform,
                    epochs=epochs,
                    resume_training=resume_training,
                    logger=logger,
                    index=index,
                    results=results,
                    cwt_times=(cwt_time_tr, cwt_time_te),
                    model_params={
                        'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 
                        'xproc_shape': xtr_proc.shape, 'name': 'Ensemble', 
                        'hidden_units': [64, 32], 
                        'model_list': [trained_models['LSTM'], trained_models['BiGRU'], trained_models['CNN']]
                    },
                    patience=es_patience,
                    plot_graph=True,
                    y_min=y_min,
                    y_max=y_max,
                    norm_target=norm_target
                )

            if extract_features != 'no':
                # Ensemble with Process Parameters
                index += 1
                ensemble_proc_input = [xtr, xtr, xtr_cwt, xtr_proc]
                ensemble_proc_val = [xval, xval, xval_cwt, xval_proc]
                ensemble_proc_test = [xte, xte, xte_cwt, xte_proc]
                
                train_model(
                    model_name='EnsembleProc',
                    create_fn=create_ensemble,
                    train_data=(ensemble_proc_input, ytr),
                    val_data=(ensemble_proc_val, yval),
                    test_data=(ensemble_proc_test, yte),
                    folder=folder,
                    run=run,
                    cwt_transform=cwt_transform,
                    epochs=epochs,
                    resume_training=resume_training,
                    logger=logger,
                    index=index,
                    results=results,
                    cwt_times=(cwt_time_tr, cwt_time_te),
                    model_params={
                        'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 
                        'xproc_shape': xtr_proc.shape, 'name': 'EnsembleProc', 
                        'hidden_units': [64, 32], 
                        'model_list': [trained_models['LSTM'], trained_models['BiGRU'], trained_models['CNN']],
                        'proc_input': True
                    },
                    patience=es_patience,
                    plot_graph=True,
                    y_min=y_min,
                    y_max=y_max,
                    norm_target=norm_target
                )

        # ResNet Variants
        if test_mode:
            resnet_configs = []
        else:
            resnet_configs = [
                {
                    'name': 'ResNet',
                    'create_fn': create_resnet,
                    'params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'regress': True, 'attention': False, 'scalogram_cwt': cwt_transform},
                    'data': (xtr_cwt, xval_cwt, xte_cwt),
                    'patience': es_patience,
                    'plot_graph': True,
                    'epochs': 500
                },
                {
                    'name': 'RobustResNet',
                    'create_fn': create_resnet,
                    'params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'regress': True, 'attention': True, 'scalogram_cwt': cwt_transform},
                    'data': (xtr_cwt, xval_cwt, xte_cwt),
                    'patience': es_patience,
                    'plot_graph': True,
                    'epochs': 500
                }
            ]

        for config in resnet_configs:
            index += 1
            train_model(
                model_name=config['name'],
                create_fn=config['create_fn'],
                train_data=(config['data'][0], ytr),
                val_data=(config['data'][1], yval),
                test_data=(config['data'][2], yte),
                folder=folder,
                run=run,
                cwt_transform=cwt_transform,
                epochs=config.get('epochs', epochs),
                resume_training=resume_training,
                logger=logger,
                index=index,
                results=results,
                cwt_times=(cwt_time_tr, cwt_time_te),
                model_params=config['params'],
                patience=config['patience'],
                plot_graph=config.get('plot_graph', False),
                y_min=y_min,
                y_max=y_max,
                norm_target=norm_target
            )

        proc_configs = [
            {
                'name': 'CNNProc',
                'base_fn': create_cnn,
                'input_type': 'cwt',
                'base_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'scalogram_cwt': cwt_transform},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'CNNProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience
            },
            {
                'name': 'LSTMProc',
                'base_fn': create_lstm,
                'input_type': 'raw',
                'base_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr.shape},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'LSTMProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience_lstm,
                'epochs': 500
            },
            {
                'name': 'BiGRUProc',
                'base_fn': create_bigru,
                'input_type': 'raw',
                'base_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr.shape},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'BiGRUProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience_lstm,
                'epochs': 500
            },
            {
                'name': 'ResNetProc',
                'base_fn': create_resnet,
                'input_type': 'cwt',
                'base_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'regress': False, 'attention': False, 'scalogram_cwt': cwt_transform},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'ResNetProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience
            },
            {
                'name': 'RobustResNetProc',
                'base_fn': create_resnet,
                'input_type': 'cwt',
                'base_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'x_shape': xtr_cwt.shape, 'regress': False, 'attention': True, 'scalogram_cwt': cwt_transform},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'RobustResNetProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience
            },
            {
                'name': 'TransformerProc',
                'base_fn': create_transformer,
                'input_type': 'raw',
                'base_params': {'learning_r': 1e-4, 'decay': 1e-5, 'dropout': 0.1, 'x_shape': xtr.shape, 'num_heads': 4, 'embed_dim': 64, 'num_blocks': 4, 'scalogram_cwt': None},
                'ensemble_params': {'learning_r': 1e-3, 'decay': 1e-5, 'dropout': 0.2, 'xproc_shape': xtr_proc.shape, 'name': 'TransformerProc', 'hidden_units': [64, 32], 'proc_input': True, 'freeze_base': False},
                'patience': es_patience,
                'epochs': 500
            }
        ]
        if extract_features == 'no':
            proc_configs = [config for config in proc_configs if 'ResNet' in config['name']]
        elif test_mode:
            proc_configs = [config for config in proc_configs if config['name'] == 'CNNProc']

        for config in proc_configs:
            index += 1
            
            # Construct merged params
            merged_params = {
                'base_model_fn': config['base_fn'],
                'base_model_params': config['base_params'],
                'ensemble_params': config['ensemble_params']
            }
            
            # Select data based on input_type
            if config['input_type'] == 'cwt':
                x_train = [xtr_cwt, xtr_proc]
                x_val = [xval_cwt, xval_proc]
                x_test = [xte_cwt, xte_proc]
            else: # raw
                x_train = [xtr, xtr_proc]
                x_val = [xval, xval_proc]
                x_test = [xte, xte_proc]

            train_model(
                model_name=config['name'],
                create_fn=create_proc_model,
                train_data=(x_train, ytr),
                val_data=(x_val, yval),
                test_data=(x_test, yte),
                folder=folder,
                run=run,
                cwt_transform=cwt_transform,
                epochs=config.get('epochs', 500), # Default to 500 if not specified
                resume_training=resume_training,
                logger=logger,
                index=index,
                results=results,
                cwt_times=(cwt_time_tr, cwt_time_te),
                model_params=merged_params,
                patience=config['patience'],
                plot_graph=True,
                y_min=y_min,
                y_max=y_max,
                norm_target=norm_target
            )

        xtr_cwt = None
        xval_cwt = None
        xte_cwt = None

    df = pd.DataFrame(results).drop(['history'], axis=1)
    df.to_csv(f'{folder}/DL_{run}_scores.csv',sep=';',decimal='.')
    logger.info(df)

    logger.info('================== Training finished')

    logger.info(f"---Script time\n--- {(time.time() - script_time):.4f} seconds ---")
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()

    # Memory cleanup
    del xtr, xval, xte, xtr_cwt, xval_cwt, xte_cwt, trained_models, results, nasa_train_base, nasa_val_base, nasa_test_base
    try:
        del model
    except:
        pass
    
    K.clear_session()
    gc.collect()

if __name__ == "__main__":
    
    epochs=500
    sliding_window_size=250
    sliding_window_stride=125
    resume_training=True
    test_mode=False
    norm_target=[False]
    runs_per_case_test_val=1
    split_offsets=[2, 5]
    use_generative = [False, True]
    extract_features='yes'

    print('Args received: ', sys.argv[1:])

    opts, args = getopt.getopt(sys.argv[1:],"he:p:c:r:t")
    for opt, arg in opts:
      if opt == '-h':
        print ('DL_manual_regression.py\n\t-a <apply_averaging (1 or 0)>\n\t-e <epochs>\n\t-w <average_window_size>\n\t-p <preload_model_file>\n\t-r <resume_training (1 or 0)>\n\t-t <test_mode>')
        sys.exit()
      elif opt in ("-e"):
        epochs = int(arg)
      elif opt in ("-r"):
        resume_training = arg == "1"
      elif opt in ("-t"):
        test_mode = True
      elif opt in ("-c"):
        use_generative = [arg == "1"]

    if resume_training:
        print(f'================== Resuming training')
    else:
        print(f'================== Starting new training')
        
    print(f'================== Training for {epochs} epochs')
    print(f'================== Augmenting data with a sliding window size of {sliding_window_size} and a stride of {sliding_window_stride}')

    print('Press any key to begin')
    inp = input()

    seed = 42

    print(f'================== Training with seed {seed}')
    for nt in norm_target:
        print(f'================== Normalisation for target values: {nt}')
        for cvae in use_generative:
            print(f'================== Generative: {cvae}')
            for offset in split_offsets:
                print(f'================== Split offset: {offset}')
                for run_name in ['all','AC_table','AC','DC','DC_table','ACDC']:
                    print(f'================== Training {run_name}')
                    main(epochs=epochs, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                        run=run_name, resume_training=resume_training, seed=seed, test_mode=test_mode, norm_target=nt,
                        runs_per_case_test_val=runs_per_case_test_val, split_offset=offset, 
                        use_generative=cvae)
