#https://autokeras.com/
#https://autokeras.com/tutorial/faq/#how-to-resume-a-previously-killed-run

#https://pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/

import sys, getopt
import glob

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#os.environ["KERAS_BACKEND"] = "tensorflow" #Available backend options are: "jax", "tensorflow", "torch"
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from joblib import dump


from keras.models import Model, load_model
from keras.utils import plot_model
from keras import Sequential
from keras.layers import Input, LSTM, Dense, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Bidirectional, GRU, concatenate, Dropout, BatchNormalization, Activation, Add, Multiply
from keras.optimizers import Adam, SGD
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.ops import squeeze, sqrt, mean, square

from helpers import compute_mean_std
from data import NASA_Dataset, StdScalerTransform, MinMaxScalerTransform
from skimage.transform import resize

from keras import backend as K

import json
import time

import logging
import datetime

sns.set_context("paper")
sns.set(font_scale = 1)
bgcolor=''
color=['hls','Plasma']

data_transform_cwt = [None]#['cmor',None]
cwt_dir = 'CWT_images'
wavelet = 'morse'

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

def conv_bn_relu(x, filters, kernel_size, padding='valid', name='', line=0, decay=1e-5, signal_qty=1, is_2d=False):
    if signal_qty == 1 and not is_2d:
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=l2(decay), name=name+'_Conv_'+str(line))(x)
    #elif signal_qty > 1 and is_2d:
    #    x = Conv3D(filters=filters, kernel_size=(kernel_size, kernel_size, kernel_size), padding=padding, kernel_regularizer=l2(decay), name=name+'_Conv_'+str(line))(x)
    else:
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, kernel_regularizer=l2(decay), name=name+'_Conv_'+str(line))(x)
        
    x = BatchNormalization(name=name+'_BN_'+str(line))(x)
    x = Activation('relu', name=name+'_ReLU_'+str(line))(x)
    return x

def resnet_block(x, filters, kernel_size, name='', line=0, attention=False, downsample=False, signal_qty=1, is_2d=False):
    shortcut = x
    x = conv_bn_relu(x, filters, kernel_size, padding='same', name=name, line=line, signal_qty=signal_qty, is_2d=is_2d)
    x = conv_bn_relu(x, filters, kernel_size, padding='same', name=name, line=str(line)+'_1', signal_qty=signal_qty, is_2d=is_2d)
    if downsample:
        if signal_qty == 1 and not is_2d:
            shortcut = Conv1D(kernel_size=1, strides=1, filters=filters, padding="same", name=name+'_res_'+str(line)+'_down')(shortcut)
        #elif signal_qty > 1 and is_2d:
        #    shortcut = Conv3D(kernel_size=1, strides=1, filters=filters, padding="same", name=name+'_res_'+str(line)+'_down')(shortcut)
        else:
            shortcut = Conv2D(kernel_size=1, strides=2, filters=filters, padding="same", name=name+'_res_'+str(line)+'_down')(shortcut)
    if attention:
        x = Multiply(name=name+f'_mult_{line}')([x, shortcut])
    x = Add(name=name+f'_add_{line}')([x, shortcut])
    return x

def feature_denoising(x, filters, kernel_size, dropout, signal_qty=1, name='', padding='valid'):
    if signal_qty == 1:
        x = Conv1D(filters=filters, kernel_size=5, name=name+'_Denoise')(x)
    else:
        x = Conv2D(filters=filters, kernel_size=(5, 1), name=name+'_Denoise')(x)
    x = BatchNormalization(name=name+'_Denoise_BN')(x)
    x = Activation('relu', name=name+'_Denoise_ReLU')(x)
    x = Dropout(dropout, name=name+'_Denoise_drop')(x)
    return x

def root_mse(y_test, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_test)))  #mean_squared_error(y_test, y_pred,squared=False)

def get_scores(y_test, y_pred):
    rmse = float(tf.squeeze(root_mse(y_test, y_pred)).numpy())
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return rmse, r2, mae, mape

def evaluate_save_model(model, model_name, index, x_test, y_test, run_name, results, model_results, folder=None):
    # Evaluate the model on the test set
    start_time = time.time()
    ypred = model.predict(x_test)
    total_time = time.time() - start_time
    model_results['eval_time'] = total_time
    
    rmse, r2, mae, mape = get_scores(y_test, ypred)
    print(f'RMSE: {rmse:.5f}\nR2: {r2:.5f}\nMAE: {mae:.5f}\nMAPE: {mape:.5f}')
    
    file_name = f'{folder + "/" if folder is not None else ""}DL_{model_name}_{run_name}_{rmse:.4f}.h5'

    model_results['model_file'] = file_name
    model_results['r2_score'] = r2
    model_results['rmse'] = rmse
    results.append(model_results)

    model.save(file_name)

    results_df = pd.DataFrame({'Ground truth' : np.squeeze(y_test), 'Prediction' : np.squeeze(ypred)},columns=['Ground truth', 'Prediction']).sort_values(by=['Ground truth'], ignore_index=True)

    plt.plot(results_df['Prediction'], label='Prediction')
    plt.plot(results_df['Ground truth'], label='Ground truth')
    plt.legend(loc='upper left')
    plt.savefig(f"{folder + '/' if folder is not None else ''}/DL_{model_name}_{run_name}_test_pred.png")
    plt.clf()

def create_lstm(learning_r=1e-4,
                decay=1e-5,
                dropout=0.1,
                x_shape=[1,1],
                name='LSTM',
                hidden_units=[256,256]):
    signal_input_lstm = Input((x_shape[1], x_shape[2]), name=name+'_signal_input')  
    x = LSTM(hidden_units[0], return_sequences=True, kernel_regularizer=l2(decay), name=name+'_1')(signal_input_lstm)
    x = BatchNormalization(name=name+'_BN_1')(x)
    for i in range(1, len(hidden_units)):
        if i == len(hidden_units)-1: #Last, no return_sequences
            x = LSTM(hidden_units[i], dropout=dropout, kernel_regularizer=l2(decay), name=name+f'_{i+1}')(x)
        else: #Rest, no L2
            x = LSTM(hidden_units[i], return_sequences=True, dropout=dropout, name=name+f'_{i+1}')(x)
        x = BatchNormalization(name=name+f'_BN_{i+1}')(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(decay), name=name+'_dense')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_lstm = Model(inputs=signal_input_lstm, outputs=x, name=name)    
    model_lstm.compile(loss=root_mse, optimizer=Adam(learning_rate=learning_r))

    return model_lstm

def create_bigru(learning_r=1e-4,
                decay=1e-5,
                dropout=0.1,
                x_shape=[1,1],
                name='BiGRU',
                hidden_units=[256,256]):
    signal_input_bigru = Input((x_shape[1], x_shape[2]), name=name+'_signal_input')  
    
    x = Bidirectional(GRU(hidden_units[0], return_sequences=True, kernel_regularizer=l2(decay)), name=name+'_1')(signal_input_bigru)
    x = BatchNormalization(name=name+'_BN_1')(x)
    for i in range(1, len(hidden_units)):
        if i == len(hidden_units)-1: #Last, no return_sequences
            x = Bidirectional(GRU(hidden_units[i], dropout=dropout, kernel_regularizer=l2(decay)), name=name+f'_{i+1}')(x)
        else: #Rest, no L2
            x = Bidirectional(GRU(hidden_units[i], return_sequences=True, dropout=dropout), name=name+f'_{i+1}')(x)
        x = BatchNormalization(name=name+f'_BN_{i+1}')(x)
        
    x = Dense(64, activation='relu', kernel_regularizer=l2(decay), name=name+'_dense')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_bigru = Model(inputs=signal_input_bigru, outputs=x, name=name)    
    model_bigru.compile(loss=root_mse, optimizer=Adam(learning_rate=learning_r))

    return model_bigru

def create_cnn(learning_r=1e-4,
               decay=1e-5,
               dropout=0.1,
               x_shape=[1,1],
               name='CNN',
               filters=[64, 32],
               kernel_size=[4, 2],
               pooling=[4, 2],
               apply_dropout=[True, True],
               apply_pooling=[True, True],
               scalogram_cwt=None):

    is_2d = scalogram_cwt is not None
    padding = 'valid' if x_shape[2] == 1 else 'same'
    signal_qty = x_shape[2]
    
    signal_input_cnn = Input((x_shape[1], x_shape[2], 1) if x_shape[2] > 1 and not is_2d else (x_shape[1], x_shape[2]), name=name+'_signal_input')
    
    # Insert CWT preprocessor Tensor layer if requested
    #if is_2d:
    #    x = CWTLayer(wavelet     = wavelet,
    #                 scales      = 224,          # ← height of the scalogram == 224
    #                 kernel_length_factor = 10,     # kernel ≈ 10 × largest scale
    #                 output_size = (224, x_shape[1]))(signal_input_cnn)
    #    padding = 'same'  # 2D CNN path

    x = conv_bn_relu(x if is_2d else signal_input_cnn, filters[0], kernel_size[0], name=name, line=1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d)
    if apply_dropout[0]:
        x = Dropout(dropout, name=name+'_drop_1')(x)
    if apply_pooling[0]:
        if x_shape[2] == 1 and not is_2d:
            x = MaxPooling1D(pooling[0], name=name+'_pool_1')(x)
        #elif x_shape[2] > 1 and is_2d:
        #    x = MaxPooling3D(pool_size=(pooling[0], pooling[0], pooling[0]), name=name+'_pool_1', padding=padding)(x)
        else:
            x = MaxPooling2D(pool_size=pooling[0], name=name+'_pool_1', padding=padding)(x)
    
    for i in range(1, len(filters)):
        x = conv_bn_relu(x, filters[i], kernel_size[i], name=name, line=i+1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d)
        if apply_dropout[i]:
            x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
        if apply_pooling[i]:
            if x_shape[2] == 1 and not is_2d:
                x = MaxPooling1D(pool_size=pooling[i], name=name+f'_pool_{i+1}')(x)
            #elif x_shape[2] > 1 and is_2d:
            #    x = MaxPooling3D(pool_size=(pooling[i], pooling[i], pooling[i]), name=name+f'_pool_{i+1}', padding=padding)(x)
            else:
                x = MaxPooling2D(pool_size=pooling[i], name=name+f'_pool_{i+1}', padding=padding)(x)
        
    x = Flatten(name=name+'_flatten')(x)
    x = Dense(128, activation='relu', name=name+'_dense')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    model_cnn = Model(inputs=signal_input_cnn, outputs=x, name=name)    
    
    model_cnn.compile(loss=root_mse, optimizer=Adam(learning_rate=learning_r))
    
    return model_cnn

def create_resnet(learning_r=1e-4,
               decay=1e-5,
               dropout=0.1,
               x_shape=[1,1],
               name='resnet',
               filters=[32, 64, 64, 64, 64, 64, 128],
               kernel_size=[5, 10, 5, 5, 5, 5, 5],
               pooling=[0, 3, 3, 3, 3, 3, 3],
               apply_dropout=[True, False, False, False, False, False, False],
               apply_pooling=[False, True, False, False, False, False, True],
               is_resnet=[False, False, True, True, True, True, False],
               regress=True, 
               attention=False,
               scalogram_cwt=None):
    
    is_2d = scalogram_cwt is not None
    padding = 'valid' if x_shape[2] == 1 else 'same'
    signal_qty = x_shape[2]
    
    signal_input_resnet = Input((x_shape[1], x_shape[2], 1) if signal_qty > 1 else (x_shape[1], x_shape[2]), name=name+'_signal_input')
    x = signal_input_resnet
    
    x = feature_denoising(x, filters[0], kernel_size[0], dropout if apply_dropout[0] else 0, name=name, signal_qty=signal_qty, padding=padding)
    
    for i in range(1, len(filters)):
        if not is_resnet[i]:
            x = conv_bn_relu(x, filters[i], kernel_size[i], name=name, line=i+1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d)
        else:
            downsample = filters[i] < filters[i-1]
            x = resnet_block(x, filters[i], kernel_size[i], name=name, line=i+1, attention=attention, downsample=downsample, signal_qty=signal_qty, is_2d=is_2d)
            
        if apply_dropout[i]:
            x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
            
        if apply_pooling[i]:
            if x_shape[2] == 1 and not is_2d:
                x = MaxPooling1D(pool_size=pooling[i], name=name+f'_pool_{i+1}')(x)
            #elif x_shape[2] > 1 and is_2d:
            #    x = MaxPooling3D(pool_size=(pooling[i], pooling[i], pooling[i]), name=name+f'_pool_{i+1}', padding=padding)(x)
            else:
                x = MaxPooling2D(pool_size=pooling[i], name=name+f'_pool_{i+1}', padding=padding)(x)
        
    x = Flatten(name=name+'_flatten')(x)
    x = Dense(64, activation='relu', name=name+'_dense')(x)
    if regress:
        x = Dense(1, activation='linear', name=name+'_output')(x)
    model_resnet = Model(inputs=signal_input_resnet, outputs=x, name=name)    
    
    if regress:
        model_resnet.compile(loss=root_mse, optimizer=Adam(learning_rate=learning_r))
    
    return model_resnet

def create_ensemble(learning_r=1e-4,
                    decay=1e-5,
                    dropout=0.1,
                    xproc_shape=[1,1],
                    name='ens',
                    hidden_units=[32],
                    model_list=[],
                    proc_input=False,
                    freeze_base=True):# update all layers in all base learners to not be trainable
    
    if len(model_list) > 0:
        if freeze_base:
            for i in range(len(model_list)):
                model = model_list[i]
                for layer in model.layers:
                    # make not trainable
                    layer.trainable = False
                    #rename to avoid 'unique layer name' issue
                    #layer._name = 'ensemble_' + str(i+1) + '_' + layer.name

        if proc_input:
            process_input = Input(shape=(xproc_shape[1],), name='process_input')
            y = Dense(xproc_shape[1], activation='linear', name='process_output')(process_input)
            model_process_input = Model(inputs=process_input, outputs=y, name='process_data')

        # define multi-headed input concatenating inputs from all models
        ensemble_inputs = [model.input for model in model_list]
        if proc_input:
            ensemble_inputs.append(model_process_input.input)
        # concatenate output from all models
        ensemble_outputs = [model.output for model in model_list]
        if proc_input:
            ensemble_outputs.append(model_process_input.output)

        #merge outputs of the models
        merge = concatenate(ensemble_outputs, name=name+'_concat')
        x = BatchNormalization(name=name+'_concat_BN')(merge)
        
        for i in range(len(hidden_units)):
            x = Dense(hidden_units[i], activation='relu', kernel_regularizer=l2(decay), name=name+f'_{i+1}')(x)
            if dropout != 0:
                x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
            x = BatchNormalization(name=name+f'_BN_{i+1}')(x)
            
        x = Dense(1, activation='linear', name=name+'_output')(x)
        model_ens = Model(inputs=ensemble_inputs, outputs=x)

        model_ens.compile(loss=root_mse, optimizer=Adam(learning_rate=learning_r))
        return model_ens
    else:
        return None

def main(epochs=300, 
        sliding_window_size=250,
        sliding_window_stride=25,
        run = '',
        resume_training=False):

    results = []
    model_list = []
    
    os.makedirs(cwt_dir, exist_ok=True)

    folder = 'manual'
    if not os.path.exists(f'{folder}/DL_{run}_scores.csv') or not resume_training:

        # Get the current date and time
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
        
        # Create a logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
    
        # Create a file handler
        handler = logging.FileHandler(f'manual/manual_{timestamp}.log')
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

        split_ratios = (0.48, 0.12, 0.4) 
        # Compute mean and std for the training data
        train_dataset = NASA_Dataset(split='train', signal_group=run, split_ratios=split_ratios)
        mean, std = compute_mean_std(train_dataset)
        # Define normalisation transform
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
        index = 0          
            
        for cwt_transform in data_transform_cwt:
            model_name = "CNN"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_cnn = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_cnn = create_cnn(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape, scalogram_cwt=cwt_transform)
        
                # Feed the image classifier with training data.
                history = model_cnn.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_cnn = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_cnn, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_cnn)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "LSTM"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_lstm = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_lstm = create_lstm(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape)
        
                # Feed the image classifier with training data.
                history = model_lstm.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_lstm = load_model('best_model.h5', compile=False)            
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_lstm, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_lstm)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "BiGRU"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_gru = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_gru = create_bigru(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape)
        
                # Feed the image classifier with training data.
                history = model_gru.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_gru = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_gru, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_gru)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
                        
            model_name = "Ensemble"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_ens = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
            
                model_ens = create_ensemble(learning_r=1e-4,
                                decay=1e-5,
                                dropout=0.1,
                                xproc_shape=xtr_proc.shape,
                                name=model_name,
                                hidden_units=[64, 32],
                                model_list=[model_lstm, model_gru, model_cnn])
                # plot graph of ensemble
                plot_model(model_ens, show_shapes=True, to_file=f'{folder}/DL_ensemble_graph_{run}_{cwt_transform}.png')
                
                X = [xtr if i != 1 or xtr.shape[2] == 1 else np.expand_dims(xtr, axis=-1) for i in range(len(model_ens.input))]
                #X.append(xtr_proc_e)
                X_VAL = [xval if i != 1 or xtr.shape[2] == 1 else np.expand_dims(xval, axis=-1) for i in range(len(model_ens.input))]
                #X_VAL.append(xval_proc_e)
        
                history = model_ens.fit(X, ytr, batch_size=32, epochs=epochs, validation_data=(X_VAL, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_ens = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            X = [xte if i != 1 or xtr.shape[2] == 1 else np.expand_dims(xte, axis=-1) for i in range(len(model_ens.input))]
            #X.append(xte_proc_e)
            evaluate_save_model(model_ens, model_name, index, X, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_ens)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "ResNet"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_resnet = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_resnet = create_resnet(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape, regress=True, attention=False, scalogram_cwt=cwt_transform)
        
                # plot graph of ensemble
                plot_model(model_resnet, show_shapes=True, to_file=f'{folder}/DL_{model_name}_graph_{run}_{cwt_transform}.png')
                
                # Feed the image classifier with training data.
                history = model_resnet.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_resnet = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_resnet, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_resnet)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "RobustResNet"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_robresnet = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_robresnet = create_resnet(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape, regress=True, attention=True, scalogram_cwt=cwt_transform)
        
                # plot graph of ensemble
                plot_model(model_robresnet, show_shapes=True, to_file=f'{folder}/DL_{model_name}_graph_{run}_{cwt_transform}.png')
                
                # Feed the image classifier with training data.
                history = model_robresnet.fit(xtr, ytr, batch_size=32, epochs=epochs, validation_data=(xval, yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_robresnet = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_robresnet, model_name, index, xte, yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_robresnet)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "ResNetProc"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_ens_res = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_resnet_tmp = create_resnet(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape, regress=False, attention=False, scalogram_cwt=cwt_transform)
                model_ens_res = create_ensemble(learning_r=1e-4,
                                decay=1e-5,
                                dropout=0.1,
                                xproc_shape=xtr_proc.shape,
                                name=model_name,
                                hidden_units=[64, 32],
                                model_list=[model_resnet_tmp], 
                                proc_input=True,
                                freeze_base=False)
                # plot graph of ensemble
                plot_model(model_ens_res, show_shapes=True, to_file=f'{folder}/DL_{model_name}_graph_{run}_{cwt_transform}.png')
        
                # Feed the image classifier with training data.
                history = model_ens_res.fit([xtr, xtr_proc], ytr, batch_size=32, epochs=epochs, validation_data=([xval, xval_proc], yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_ens_res = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_ens_res, model_name, index, [xte, xte_proc], yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_ens_res)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
            
            model_name = "RobustResNetProc"
            print(f'\n================== Training {model_name} with {cwt_transform} as CWT transform')
            index += 1
            es = None
            mc = None
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100, min_delta=0.0001)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            start_time = time.time()
            model_results = {
                'model_name' : model_name,
                'scalogram' : str(cwt_transform)
            }
        
            print(f'Look for file {folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            matches = glob.glob(f'{folder}/DL_{model_name}_{run}_{cwt_transform}_*.h5')
            if len(matches) > 0 and resume_training:
                logger.info(f'================== Skipping training of {model_name} for run {run} - {matches[0]} exists already, loading file...')
                model_ens_robres = load_model(matches[0], compile=False)
                with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json') as fp:
                    model_results = json.load(fp)
                total_time = model_results['time']
                logger.info(f'================== Model {model_name} for run {run} loaded')
            else:
                logger.info(f'================== Training {model_name} for run {run}')
                model_robresnet_tmp = create_resnet(learning_r=1e-4, decay=1e-5, dropout=0.1, x_shape=xtr.shape, regress=False, attention=True, scalogram_cwt=cwt_transform)
                model_ens_robres = create_ensemble(learning_r=1e-4,
                                decay=1e-5,
                                dropout=0.1,
                                xproc_shape=xtr_proc.shape,
                                name=model_name,
                                hidden_units=[64, 32],
                                model_list=[model_robresnet_tmp], 
                                proc_input=True,
                                freeze_base=False)
                # plot graph of ensemble
                plot_model(model_ens_robres, show_shapes=True, to_file=f'{folder}/DL_{model_name}_graph_{run}_{cwt_transform}.png')
        
                # Feed the image classifier with training data.
                history = model_ens_robres.fit([xtr, xtr_proc], ytr, batch_size=32, epochs=epochs, validation_data=([xval, xval_proc], yval), verbose=1, callbacks=[es, mc])
                logger.info(f'================== {model_name} Model history')
                logger.info(history.history.keys())
                plt.plot(history.history['loss'], label='Training')
                plt.plot(history.history['val_loss'], label='Validation')
                plt.legend()
                plt.savefig(f"{folder}/DL_{model_name}_{run}_{cwt_transform}_training_loss_{run}_{cwt_transform}.png")
                plt.clf()
                model_ens_robres = load_model('best_model.h5', compile=False)       
                total_time = time.time() - start_time
                model_results['time'] = total_time
                model_results['history'] = history.history
                
            evaluate_save_model(model_ens_robres, model_name, index, [xte, xte_proc], yte, f'{run}_{cwt_transform}', results, model_results, folder=folder)
            with open(f'{folder}/DL_{model_name}_{run}_{cwt_transform}.json', "w") as outfile:
                outfile.write(json.dumps(model_results, indent=4))
                
            model_list.append(model_ens_robres)
            logger.info("--- %s seconds ---" % (total_time))
            logger.info(f'================== {model_name} model trained')
    
        df = pd.DataFrame(results).drop(['history'], axis=1)
        df.to_csv(f'{folder}/DL_{run}_scores.csv',sep=';',decimal='.')
        logger.info(df)
    
        logger.info('================== Training finished')
    
        logger.info(f"---Script time\n--- {(time.time() - script_time):.4f} seconds ---")

if __name__ == "__main__":
    
    epochs=1000
    sliding_window_size=500
    sliding_window_stride=50
    resume_training=True

    print('Args received: ', sys.argv[1:])

    opts, args = getopt.getopt(sys.argv[1:],"he:p:c:r:")
    for opt, arg in opts:
      if opt == '-h':
        print ('DL_manual_regression.py\n\t-a <apply_averaging (1 or 0)>\n\t-e <epochs>\n\t-w <average_window_size>\n\t-p <preload_model_file>\n\t-r <resume_training (1 or 0)>')
        sys.exit()
      elif opt in ("-e"):
        epochs = int(arg)
      elif opt in ("-r"):
        resume_training = arg == "1"

    if resume_training:
        print(f'================== Resuming training')
    else:
        print(f'================== Starting new training')
        
    print(f'================== Training for {epochs} epochs')
    print(f'================== Augmenting data with a sliding window size of {sliding_window_size} and a stride of {sliding_window_stride}')

    print('Press any key to begin')
    inp = input()

    for run_name in ['AC_table','AC','DC','DC_table','ACDC','all']:
        print(f'================== Training {run_name}')
        main(epochs=epochs, sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
            run=run_name, resume_training=resume_training)