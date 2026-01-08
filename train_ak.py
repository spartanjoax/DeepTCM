
import sys, getopt
import glob
import os

# Environment Setup
os.environ["KERAS_BACKEND"] = "torch"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc

import json
import time
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

import keras
import keras_tuner as kt
from keras.models import load_model, save_model
from keras.utils import plot_model

# Project Imports
from helpers import compute_mean_std, evaluate_model
from data import NASA_Dataset, StdScalerTransform, MinMaxScalerTransform
from models.autokeras_blocks import AutoKerasHyperModel
from models.preprocessing_tuner import PreprocessingTuner
from models.generative import train_generative_model, generate_synthetic_data

from helpers import apply_moving_average, apply_high_pass_filter, apply_low_pass_filter, apply_detrend, apply_rms
from helpers import convert_to_cwt, convert_to_fsst, convert_to_wsst

split_type='runs'

# Logging Setup
def setup_logging(timestamp):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'AutoML_AK_search_{timestamp}.log')
    handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    return logger



def main(epochs=1, 
        sliding_window_size=250,
        sliding_window_stride=125,
        num_outer_trials=20,
        num_inner_trials=50,
        run_name = '',
        resume_training=False,
        runs_per_case_test_val=1,
        split_offset=5,
        seed=42,
        ma_window_size=10,
        gen_norm_type='minmax'):
    
    # Force Garbage Collection and GPU cache clearing
    keras.utils.clear_session()
    gc.collect()
    torch.cuda.empty_cache()
    
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
    logger = setup_logging(timestamp)
    
    folder = 'automl'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # --- Data Loading (Reused from train.py) ---
    logger.info("Loading Data...")
    
    # Process Parameters Min/Max
    proc_min = [0.1, 0.05, 1, 50] 
    proc_max = [1.5, 0.5, 2, 200]
    minmax_transform = MinMaxScalerTransform(min=proc_min, max=proc_max)
    
    run_id = run_name[1:] # e.g. '_ACDC' -> 'ACDC'
    split_ratios = (0.48, 0.12, 0.4) 
    
    # Training Data Mean/Std
    train_dataset = NASA_Dataset(split='train', signal_group=run_id, split_ratios=split_ratios, 
                                sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                                seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, 
                                split_offset=split_offset, apply_averaging=True, avg_window_size=ma_window_size)
    mean, std = compute_mean_std(train_dataset)
    normalize_transform = StdScalerTransform(mean=mean, std=std)

    # Load Full Datasets
    nasa_train = NASA_Dataset(split='train', signal_group=run_id, split_ratios=split_ratios, 
                            transformX=normalize_transform, transformProc=minmax_transform, 
                            sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                            seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset, 
                            apply_averaging=False)
    nasa_val = NASA_Dataset(split='val', signal_group=run_id, split_ratios=split_ratios, 
                            transformX=normalize_transform, transformProc=minmax_transform, 
                            sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                            seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset, 
                            apply_averaging=False)
    nasa_test = NASA_Dataset(split='test', signal_group=run_id, split_ratios=split_ratios, 
                            transformX=normalize_transform, transformProc=minmax_transform, 
                            sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride,
                            seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset, 
                            apply_averaging=False)

    xtr, xtr_proc, ytr = nasa_train.data.cpu().data.numpy(), nasa_train.proc_data.cpu().data.numpy(), nasa_train.targets.cpu().data.numpy()
    xval, xval_proc, yval = nasa_val.data.cpu().data.numpy(), nasa_val.proc_data.cpu().data.numpy(), nasa_val.targets.cpu().data.numpy()
    xte, xte_proc, yte = nasa_test.data.cpu().data.numpy(), nasa_test.proc_data.cpu().data.numpy(), nasa_test.targets.cpu().data.numpy()
    
    logger.info(f"Train info: X={xtr.shape}, Proc={xtr_proc.shape}, Y={ytr.shape}")
    
    # --- Ensure Generative Models are Trained ---
    output_dir = 'generative_results'
    gen_models_list = {'ConditionalVAE':200, 'TimeGAN':200}

    # Train on 'all' signal group for general capability, matching train.py logic
    # We need a temp dataset for this
    gen_train_ds = NASA_Dataset(split='train', signal_group='all', 
                                sliding_window_size=sliding_window_size, 
                                sliding_window_stride=sliding_window_stride,
                                seed=seed,split_type=split_type, runs_per_case_test_val=runs_per_case_test_val, split_offset=split_offset, 
                                apply_averaging=True, avg_window_size=ma_window_size)
    
    for gen_name in gen_models_list.keys():
        suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
        gen_weights_path = os.path.join(output_dir, f"{gen_name}_{suffix}_ch5.weights.h5")
        
        if not os.path.exists(gen_weights_path):
             logger.info(f"Training Generative Model: {gen_name}")
                                         
             train_generative_model(gen_name, gen_train_ds, 
                                    sliding_window_size, sliding_window_stride,
                                    epochs=gen_models_list[gen_name], 
                                    output_dir=output_dir, normalization_type=gen_norm_type)
        else:
             logger.info(f"Found existing weights for {gen_name}")

    # --- AutoKeras Configuration ---

    config = {
        "project_name": "preproc", # Default base name, will be overwritten by tuner
        "directory": f"{folder}/{run_id}",
        "autokeras": {
            "max_trials": num_inner_trials,
            "objective": "val_loss",
            "overwrite": not resume_training
        }
    }
    
    hypermodel = AutoKerasHyperModel(config)
    hypermodel.set_input_shape(xtr.shape[1:])
    hypermodel.set_multimodal(True) 
    
    # --- Preprocessing Tuner Optimization (Outer Loop) ---
    logger.info("Starting Preprocessing Optimization...")
    
    # Define search space upfront for console display
    hp = kt.HyperParameters()
    hp.Choice("noise_reduction", ["moving_average", "high_pass", "low_pass", "rms"], default="moving_average")
    hp.Choice("detrend", ["none", "detrend"], default="none")
    hp.Choice("scalogram", ["none", "cwt", "fsst", "wsst"], default="none")
    hp.Choice("gen_model_type", ["none", "ConditionalVAE"], default="ConditionalVAE")#, "TimeGAN"], default="ConditionalVAE")

    # Outer Search Directory: automl/preprocessing_{run_id}
    tuner = PreprocessingTuner(
        hypermodel=hypermodel,
        hyperparameters=hp,
        objective=kt.Objective("rmse", direction="min"),
        max_trials=num_outer_trials,
        directory=f"{folder}/{run_id}",
        project_name=f"preprocessing",
        overwrite=not resume_training,
    )
    
    # Pass generative config to run_trial
    # We need mean/std from the current run to adapt generated data
    # mean, std were computed at the start
    gen_config = {
        'output_dir': 'generative_results',
        'sliding_window_size': sliding_window_size,
        'sliding_window_stride': sliding_window_stride,
        'current_mean': mean.numpy() if hasattr(mean, 'numpy') else mean,
        'current_std': std.numpy() if hasattr(std, 'numpy') else std,
        'normalization_type': 'minmax'
    }

    start_time = time.time()

    # Define train inputs
    train_input = [xtr, xtr_proc] if xtr_proc is not None else xtr
    val_input = [xval, xval_proc] if xval_proc is not None else xval
    
    # Define test data structure matching train (Signal vs [Signal, Info])
    test_input = [xte, xte_proc] if xte_proc is not None else xte
    
    tuner.search(
        x=train_input, 
        y=ytr,
        validation_data=(val_input, yval),
        test_data=(test_input, yte),
        epochs=epochs,
        gen_config=gen_config,
        ma_window_size=ma_window_size
    )

    total_time = time.time() - start_time
    
    # --- Results Export ---
    logger.info("Optimization Completed. Exporting Results...")
    
    best_hp_outer = tuner.get_best_hyperparameters()[0]
    logger.info(f"Best Preprocessing HPs: {best_hp_outer.values}")
    
    # Identify the winning inner project
    best_trial = tuner.oracle.get_best_trials(1)[0]
    logger.info(f"Best Inner Project for Trial {best_trial.trial_id}")
    
    # Use the tuner's helper method to retrieve the inner AutoModel cleanly
    clf = tuner.get_best_automodel(best_trial)
    if clf is None:
        print('No best inner trial')
        
    # User Fix: Call fit on the loaded clf to ensure it's initialized
    # We must apply the SAME PREPROCESSING to the training data that was used in the trial
    # (excluding generative augmentation which just adds samples, assume shape is consistent)
    
    x_train_pre = xtr
    y_train_pre = ytr
    x_val_pre = xval
    
    # --- 1. Generative Augmentation (Replicate best trial) ---
    gen_type = best_hp_outer.get("gen_model_type")
    
    # Need to handle info augmentation if multimodal
    x_train_info_final = None
    if xtr_proc is not None:
         x_train_info_final = xtr_proc
         
    if gen_type != "none":
        logger.info(f"Augmenting data with {gen_type} for final fit...")
        
        # Prepare condition for generation (Info + Target)
        if xtr_proc is not None:
             cond_train = np.concatenate([xtr_proc, ytr], axis=1)
        else:
             cond_train = ytr
             
        # Generate same amount as original data (100% augmentation)
        num_syn = int(x_train_pre.shape[0] * 1)
        
        syn_signal, syn_cond = generate_synthetic_data(gen_type, gen_config, num_syn, condition_data=cond_train, x_train_info=xtr_proc)
        
        if syn_signal is not None:
             x_train_pre = np.concatenate([x_train_pre, syn_signal], axis=0)
             
             # Extract y part from condition (last column)
             syn_y = syn_cond[:, -1:]
             y_train_pre = np.concatenate([y_train_pre, syn_y], axis=0)
             
             # Extract info part if multimodal
             if xtr_proc is not None:
                  syn_info = syn_cond[:, :-1]
                  x_train_info_final = np.concatenate([x_train_info_final, syn_info], axis=0)
        else:
             logger.warning("Generative augmentation returned None.")
             
    # 0. Detrend
    if best_hp_outer.get("detrend") == "detrend":
        x_train_pre = apply_detrend(x_train_pre)
        x_val_pre = apply_detrend(x_val_pre)

    # 1. Noise Reduction
    nr = best_hp_outer.get("noise_reduction")
    if nr == "moving_average":
        w = ma_window_size
        x_train_pre = apply_moving_average(x_train_pre, window_size=w)
        x_val_pre = apply_moving_average(x_val_pre, window_size=w)
    elif nr == "high_pass":
        x_train_pre = apply_high_pass_filter(x_train_pre)
        x_val_pre = apply_high_pass_filter(x_val_pre)
    elif nr == "low_pass":
        x_train_pre = apply_low_pass_filter(x_train_pre)
        x_val_pre = apply_low_pass_filter(x_val_pre)
    elif nr == "rms":
        x_train_pre = apply_rms(x_train_pre, window_size=10)
        x_val_pre = apply_rms(x_val_pre, window_size=10)

    # 2. Scalogram
    st = best_hp_outer.get("scalogram")
    if st == "cwt": 
        x_train_pre = convert_to_cwt(x_train_pre, description="CWT Re-gen for fit")
        x_val_pre = convert_to_cwt(x_val_pre, description="CWT Val Re-gen for fit")
    elif st == "fsst": 
        x_train_pre = convert_to_fsst(x_train_pre, description="FSST Re-gen for fit")
        x_val_pre = convert_to_fsst(x_val_pre, description="FSST Val Re-gen for fit")
    elif st == "wsst": 
        x_train_pre = convert_to_wsst(x_train_pre, description="WSST Re-gen for fit")
        x_val_pre = convert_to_wsst(x_val_pre, description="WSST Val Re-gen for fit")
    elif st == "stransform": 
        x_train_pre = convert_to_stransform(x_train_pre)
        x_val_pre = convert_to_stransform(x_val_pre)
    else: 
        x_train_pre = np.expand_dims(x_train_pre, axis=2)
        x_val_pre = np.expand_dims(x_val_pre, axis=2)
    
    # 3. Construct Input
    # Check if multimodal (info passed)
    # In run_trial: if x_train_info is not None: x_train_final = [x_train_signal, x_train_info]
    # Here input is xtr, xtr_proc. xtr_proc corresponds to info.
    if x_train_info_final is not None:
        x_train_final_fit = [x_train_pre, x_train_info_final]
    else:
        x_train_final_fit = x_train_pre
        
    # --- Prepare Validation Data for Fit ---
    if xval_proc is not None:
        x_val_final_fit = [x_val_pre, xval_proc]
    else:
        x_val_final_fit = x_val_pre

    logger.info("Calling clf.fit to initialize best model...")
    # We use a minimal fit call. overwrite=False in get_best_automodel should prevent retraining.
    # We pass only 1 epoch to be safe/fast if it does run.
    # Use augmented y_train_pre
    history = clf.fit(x_train_final_fit, y_train_pre, validation_data=(x_val_final_fit, yval), epochs=epochs, verbose=1)

    # Now we can access clf.tuner to get best models as in autokeras_regression.py
    results = []
    
    # Evaluate Best Models from the Inner Search
    best_model = clf.export_model()#.tuner.get_best_models(num_models=1)
    best_hps = clf.tuner.get_best_hyperparameters(num_trials=3)
    
    # Save best inner hyperparameters to JSON
    best_inner_hps_dict = {
        'time':total_time,
        'best_hps':[]
    }
    for i, hp in enumerate(best_hps):
        best_inner_hps_dict['best_hps'].append({
            "rank": i + 1,
            "values": hp.values
        })        
    with open(os.path.join(f"{folder}/{run_id}", f"best_inner_hps_{run_id}.json"), "w") as f:
        json.dump(best_inner_hps_dict, f, indent=4)
    
    # Preparation for Evaluation (Preprocessing)
    x_test_pre = xte
    x_test_proc = xte_proc
    
    # Apply Best Preprocessing to Test
    if best_hp_outer.get("detrend"):
        x_test_pre = apply_detrend(x_test_pre)
    nr = best_hp_outer.get("noise_reduction")
    if nr == "moving_average":
        #w = best_hp_outer.get("ma_window_size")
        w = ma_window_size
        x_test_pre = apply_moving_average(x_test_pre, window_size=w)
    elif nr == "high_pass":
        #c = best_hp_outer.get("hp_cutoff")
        x_test_pre = apply_high_pass_filter(x_test_pre)#, cutoff=c)
    elif nr == "low_pass":
        #c = best_hp_outer.get("lp_cutoff")
        x_test_pre = apply_low_pass_filter(x_test_pre)#, cutoff=c)
    elif nr == "rms":
        x_test_pre = apply_rms(x_test_pre, window_size=10)
    
    st = best_hp_outer.get("scalogram")
    if st == "cwt": x_test_pre = convert_to_cwt(x_test_pre, description="CWT generation for test")
    elif st == "fsst": x_test_pre = convert_to_fsst(x_test_pre, description="FSST generation for test")
    elif st == "wsst": x_test_pre = convert_to_wsst(x_test_pre, description="WSST generation for test")
    else: x_test_pre = np.expand_dims(x_test_pre, axis=2)

    logger.info(f"Evaluating Best Model")
    
    # Save Model
    file_name = f'{folder}/DL_AutoKeras_Best_{run_id}.keras'
    best_model.save(file_name)
    
    # Plot Model Architecture
    try:
        plot_model(best_model, to_file=f'{folder}/DL_AutoKeras_{run_id}_modelplot.png', show_shapes=True, show_layer_names=True)
    except Exception as e:
        logger.warning(f"Could not plot model: {e}")

    # Prepare input dynamically
    if x_test_proc is not None:
        x_eval = [x_test_pre, x_test_proc]
    else:
        x_eval = x_test_pre
            
    # Evaluate
    res = evaluate_model(
        model=best_model,
        x=x_eval,
        y=yte,
        folder=folder,
        run_name=f"AutoKeras_{run_id}_trial_{best_trial.trial_id}",
        model_results={'ModelName': f"Best"},
        history=history,
        norm_target=False
    )
    results.append(res)
    
    # Export Scores
    df = pd.DataFrame(results)
    df.to_csv(f'{folder}/DL_AutoKeras_Best_scores.csv', sep=';', decimal='.')
    
    # Export Hyperparameters (Outer)
    with open(f"{folder}/DL_AutoKeras_Best_{run_id}_outer_hp.json", "w") as f:
        json.dump(best_hp_outer.values, f, indent=4)
         
    logger.info("Finished.")
    
    # Explicit Cleanup to free GPU memory for next run
    del tuner, hypermodel, clf, best_model
    keras.utils.clear_session()
    
    # Cleanup Data
    del nasa_train, nasa_val, nasa_test, xtr, xtr_proc, ytr, xval, xval_proc, yval, xte, xte_proc, yte
    
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Simplified Argument Parsing
    epochs = 300
    resume = True
    ma_window_size=10
    gen_norm_type='minmax'
    
    # Simple CLI for demo
    opts, args = getopt.getopt(sys.argv[1:],"e:r:w:")
    for opt, arg in opts:
        if opt == '-e': epochs = int(arg)
        elif opt == '-r': resume = arg == "1"
        elif opt == '-w': ma_window_size = int(arg)
        
    # Run for specific subsets as per original script pattern
    for run_name in ['_all']:
        print(f"================== Training {run_name}")
        main(epochs=epochs, run_name=run_name, resume_training=resume, 
             ma_window_size=ma_window_size, gen_norm_type=gen_norm_type)
