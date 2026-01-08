import keras_tuner as kt
import numpy as np
import os
import torch
import gc
from tqdm import tqdm

from helpers import apply_moving_average, apply_high_pass_filter, apply_low_pass_filter, apply_detrend, apply_rms
from helpers import convert_to_cwt, convert_to_fsst, convert_to_wsst
from helpers import evaluate_model
from models.generative import generate_synthetic_data

import time

import keras
from keras import ops, callbacks
import tsgm

from joblib import load
# from data.datasets import NASA_Dataset # Assuming we might interact with dataset or passed data

class GPUMemoryCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['gpu_mem_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        logs['gpu_mem_reserved_gb'] = torch.cuda.memory_reserved() / 1e9

class TorchTensorBoard(callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard
            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(os.path.join(self._path, writer))
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        for key, value in logs.items():
            self.writer(writer).add_scalar(key, value, step)

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.add_logs("train", {k: v for k, v in logs.items() if not k.startswith("val_")}, epoch + 1)
            if isinstance(getattr(self.model, "optimizer", None), keras.optimizers.Optimizer):
                self.add_logs("train", {"learning_rate": self.model.optimizer.learning_rate}, epoch + 1)
            self.add_logs("val", {k[4:]: v for k, v in logs.items() if k.startswith("val_")}, epoch + 1)

class PreprocessingTuner(kt.BayesianOptimization):
    """
    A custom Tuner that optimizes data preprocessing steps (noise reduction, scalograms, augmentation)
    before feeding data into the AutoKeras search.
    """
    def __init__(self, hypermodel, **kwargs):
        # kt.BayesianOptimization manages the oracle internally
        super().__init__(hypermodel=hypermodel, **kwargs)
        
    def _preprocess_set(self, x_signal, noise_reduction, scalogram_type, detrend, ma_window_size=10):
        """
        Helper to preprocess a dataset given hyperparameters.
        Handles Noise Reduction and Scalograms on the signal component.
        """
        if x_signal is None:
            return None
            
        print(f"DEBUG: _preprocess_set x_signal shape: {getattr(x_signal, 'shape', 'N/A')}")
        print(f"DEBUG: HPs - nr: {noise_reduction}, st: {scalogram_type}, detrend: {detrend}")
        
        # 1. Detrend (Optional)
        if detrend == "detrend":
             x_signal = apply_detrend(x_signal)

        # 2. Noise Reduction using helpers (apply to signal)
        if noise_reduction == "moving_average":
            x_signal = apply_moving_average(x_signal, window_size=ma_window_size) # fixed window?
        elif noise_reduction == "high_pass":
            x_signal = apply_high_pass_filter(x_signal)
        elif noise_reduction == "low_pass":
            x_signal = apply_low_pass_filter(x_signal)
        elif noise_reduction == "rms":
            x_signal = apply_rms(x_signal, window_size=10)
        
        print(f"DEBUG: _preprocess_set after noise reduction x type: {type(x_signal)}")
        if isinstance(x_signal, list):
             print(f"DEBUG: x is list, len: {len(x_signal)}")
             if len(x_signal) > 0: print(f"DEBUG: x[0] type: {type(x_signal[0])}, shape: {getattr(x_signal[0], 'shape', 'N/A')}")
        else:
             print(f"DEBUG: x shape: {getattr(x_signal, 'shape', 'N/A')}")

        # 2. Scalograms (apply to signal)
        if scalogram_type == "cwt":
            x_signal = convert_to_cwt(x_signal, description="CWT generation")
        elif scalogram_type == "fsst":
            x_signal = convert_to_fsst(x_signal, description="FSST generation")
        elif scalogram_type == "wsst":
            x_signal = convert_to_wsst(x_signal, description="WSST generation")
        elif scalogram_type == "none":
            x_signal = np.expand_dims(x_signal, axis=2)
        
        print(f"DEBUG: _preprocess_set after scalogram x type: {type(x_signal)}")
        if isinstance(x_signal, list):
             print(f"DEBUG: x is list, len: {len(x_signal)}")
             if len(x_signal) > 0: print(f"DEBUG: x[0] type: {type(x_signal[0])}, shape: {getattr(x_signal[0], 'shape', 'N/A')}")
        else:
             print(f"DEBUG: x shape: {getattr(x_signal, 'shape', 'N/A')}")
        
        return x_signal

    def run_trial(self, trial, x, y, validation_data=None, **kwargs):
        """
        Custom trial execution.
        1. Defines preprocessing Hyperparameters.
        2. Applies preprocessing transformations to x (and validation_x).
        3. Builds the inner AutoKeras model.
        4. Fits the inner AutoKeras model (which runs its own NAS).
        """
        # Initialize variables for cleanup safety
        x_train_signal = None
        x_train_info = None
        x_val_signal = None
        x_val_info = None
        x_train_final = None
        x_val_final = None
        x_test_final = None
        
        x_test_signal = None
        x_test_info = None
        
        y_test = None
        
        model = None
        gen_model = None 
        best_model = None
        history = None 

        try:
            hp = trial.hyperparameters
            ma_window_size = kwargs.pop('ma_window_size', 10)
            
            # --- 1. Define Search Space ---
            
            # Noise Reduction
            noise_reduction = hp.Choice("noise_reduction", ["none", "moving_average", "high_pass", "low_pass", "rms"], default="none")
            
            # New Preprocessing Steps
            detrend = hp.Choice("detrend", ["none", "detrend"], default="none")
            
            # Scalograms
            scalogram_type = hp.Choice("scalogram", ["none", "cwt", "fsst", "wsst"], default="none")
            
            # --- 1.5 Generative Data Augmentation (Pre-preprocessing) ---
            
            # Unpack x explicitly for Generative Model (Raw Data)
            x_train_signal = None
            x_train_info = None
            if isinstance(x, list) and len(x) == 2:
                x_train_signal = x[0]
                x_train_info = x[1]
            else:
                x_train_signal = x
            
            # Search for model type
            gen_model_type = hp.Choice("gen_model_type", ["none", "ConditionalVAE", "TimeGAN"], default="ConditionalVAE")
            
            gen_time = 0
            
            print(f"Memory allocated pre-gen: {torch.cuda.memory_allocated()}")
            if gen_model_type != "none":
                start_gen = time.time()
                # Extract config
                gen_config = kwargs.get('gen_config')
                if gen_config is None:
                    print("Warning: gen_config not provided to PreprocessingTuner. Skipping generation.")
                else:
                    # Prepare condition (Info + Target)
                    if x_train_info is not None:
                         cond_train = np.concatenate([x_train_info, y], axis=1)
                    else:
                         cond_train = y
                         
                    # Generate same amount as original data (100% augmentation)
                    num_syn = int(x_train_signal.shape[0] * 1)
                    
                    # Reuse the centralized generation logic (Check generative.py for per-channel implementation)
                    syn_renorm, syn_cond = generate_synthetic_data(gen_model_type, gen_config, num_syn, condition_data=cond_train, x_train_info=x_train_info)


                    if syn_renorm is not None:
                        # Append
                        x_train_signal = np.concatenate([x_train_signal, syn_renorm], axis=0)
                        
                        # Update Info and Targets from returned conditions (which were shuffled/sampled)
                        if x_train_info is not None:
                            # syn_cond contains [Proc, Target]
                            syn_info = syn_cond[:, :-1]
                            syn_y = syn_cond[:, -1:]
                            x_train_info = np.concatenate([x_train_info, syn_info], axis=0)
                            y = np.concatenate([y, syn_y], axis=0)
                        else:
                            # syn_cond contains [Target]
                            y = np.concatenate([y, syn_cond], axis=0)
                    else:
                        print("Warning: Generative augmentation returned None or failed.")

    

            if gen_model_type != "none":
                 gen_time = time.time() - start_gen

            print(f"Memory allocated post-gen: {torch.cuda.memory_allocated()}")

            # --- 2. Apply Preprocessing (Noise Reduction/Scalograms) ---
            # Use Helper for Train and Val
            x_train_signal = self._preprocess_set(x_train_signal, noise_reduction, scalogram_type, detrend, ma_window_size=ma_window_size)
            
            x_val_signal = None
            x_val_info = None
            y_val = None
            if validation_data is not None:
                x_val_in = validation_data[0]
                y_val = validation_data[1]
                
                # Unpack Val
                if isinstance(x_val_in, list) and len(x_val_in) == 2:
                    x_val_signal = x_val_in[0]
                    x_val_info = x_val_in[1]
                else:
                    x_val_signal = x_val_in
                    
                x_val_signal = self._preprocess_set(x_val_signal, noise_reduction, scalogram_type, detrend, ma_window_size=ma_window_size)
            
            # Handle Test Data
            x_test_signal = None
            x_test_info = None
            y_test = None
            if 'test_data' in kwargs:
                test_data = kwargs['test_data']
                x_test_in = test_data[0]
                y_test = test_data[1]
                
                # Unpack Test
                if isinstance(x_test_in, list) and len(x_test_in) == 2:
                    x_test_signal = x_test_in[0]
                    x_test_info = x_test_in[1]
                else:
                    x_test_signal = x_test_in

                x_test_signal = self._preprocess_set(x_test_signal, noise_reduction, scalogram_type, detrend, ma_window_size=ma_window_size)

            # Re-pack x_train, x_val
            if x_train_info is not None:
                x_train_final = [x_train_signal, x_train_info]
            else:
                x_train_final = x_train_signal
                
            if x_val_signal is not None:
                if x_val_info is not None:
                    x_val_final = [x_val_signal, x_val_info]
                else:
                    x_val_final = x_val_signal
            else:
                x_val_final = None 
            
            x_test_final = None
            if x_test_signal is not None:
                if x_test_info is not None:
                    x_test_final = [x_test_signal, x_test_info]
                else:
                    x_test_final = x_test_signal
 
                
            kwargs_fit = kwargs.copy()
            if "gen_config" in kwargs_fit:
                kwargs_fit.pop("gen_config")
            if "test_data" in kwargs_fit:
                kwargs_fit.pop("test_data")
    
            # --- 3. Build & Fit Inner Model ---
            
            # Set dynamic project name for the inner AutoKeras search based on this trial ID
            # This isolates the NAS search for this specific set of preprocessing HPs
            if hasattr(self.hypermodel, 'set_project_name'):
                 self.hypermodel.set_project_name(f"preproc_trial_{trial.trial_id}")
            
            # The AutoKerasHyperModel.build method reads 'scalogram' from hp to decide on Block type
            model = self.hypermodel.build(hp)
            
            # # TensorBoard & GPU Monitoring Callbacks
            # log_dir = f"tensorboard_logs/{self.project_name}/trial_{trial.trial_id}"
            # tb_callback = TorchTensorBoard(path=log_dir)
            # gpu_callback = GPUMemoryCallback()
            
            # fit_callbacks = [tb_callback, gpu_callback]
            
            # # Check if user provided other callbacks (e.g. from AutoKeras) and append
            # if 'callbacks' in kwargs_fit:
            #     user_callbacks = kwargs_fit.pop('callbacks')
            #     fit_callbacks.extend(user_callbacks)

            # AutoKeras .fit returns a History object.
            # We need to handle validation data explicitely for AK
            # --- Measuse Training Time and Fit ---
            start_time = time.time()
            # AutoKeras .fit returns a History object.
            # We need to handle validation data explicitely for AK
            if x_val_final is not None and y_val is not None:
                history = model.fit(x_train_final, y, validation_data=(x_val_final, y_val), #callbacks=fit_callbacks,
                                    **kwargs_fit)
            else:
                history = model.fit(x_train_final, y, #callbacks=fit_callbacks, 
                                    **kwargs_fit)
                                    
            training_time = time.time() - start_time
            
            # --- EVALUATE AND EXPORT TRIAL RESULTS ---
            
            # Create a dictionary to store model parameters (preprocessing + others)
            model_results = {
                'trial_id': trial.trial_id,
                'scalogram_type': scalogram_type,
                'noise_reduction': noise_reduction,
                'synthetic_gen_time': gen_time
            }

            # Define folder: automl/{project_name}/trial_{id}
            # Note: self.project_name might be the generic one, we used 'preproc_trial_{id}' for inner
            trial_folder = f"{self.directory}/{self.project_name}/preproc_trial_{trial.trial_id}"
            if not os.path.exists(trial_folder):
                os.makedirs(trial_folder)
            
            # Determine appropriate validation set for evaluation
            # Use TEST data if available for final evaluation, otherwise fallback (or skip)
            
            target_x = None
            target_y = None
            run_suffix = ""
            
            if x_test_final is not None and y_test is not None:
                target_x = x_test_final
                target_y = y_test
                run_suffix = "test"
            elif x_val_final is not None and y_val is not None:
                target_x = x_val_final
                target_y = y_val
                run_suffix = "val"
                
            # Evaluate
            rmse_score = 1e9 # Default high value for failure/no-eval

            best_model = model.export_model()
            
            if target_x is not None and target_y is not None:
                eval_results = evaluate_model(
                    model=best_model,
                    x=target_x,
                    y=target_y,
                    folder=trial_folder,
                    run_name=f"trial_{trial.trial_id}_{run_suffix}",
                    history=history,
                    training_time=training_time,
                    model_results=model_results,
                    norm_target=False 
                )
                if 'rmse' in eval_results:
                    rmse_score = eval_results['rmse']
                
            # Return RMSE dictionary to Keras Tuner
            return {'rmse': rmse_score}
            
        finally:
            # Robust Cleanup
            if model is not None: del model
            if gen_model is not None: del gen_model
            if best_model is not None: del best_model
            if history is not None: del history
            
            # Delete data arrays to free memory
            del x_train_signal, x_train_info, x_val_signal, x_val_info, x_test_signal, x_test_info
            del x_train_final, x_val_final, x_test_final, y, y_val, y_test
            
            keras.utils.clear_session()
            gc.collect()
            torch.cuda.empty_cache()

    def get_best_automodel(self, trial):
        """
        Re-instantiates the AutoKeras AutoModel for a specific trial 
        to allow access to the inner tuner and best models.
        """
        project_name = f"preproc_trial_{trial.trial_id}"
        if hasattr(self.hypermodel, 'set_project_name'):
             self.hypermodel.set_project_name(project_name)
             
        # Important: Ensure we don't overwrite the existing search
        if hasattr(self.hypermodel, 'ak_config'):
            self.hypermodel.ak_config['overwrite'] = False
            
        return self.hypermodel.build()

