import os
import ntpath
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm

import scipy.io
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split

from helpers import plot_signals, plot_signals_dict, augment_signal, apply_moving_average, downsample, get_file_list, augment_dataset_softdtw, compute_min_max
import data.mat_to_csv as mtc

import torch
import tsgm
from torch.utils.data import Dataset



allowed_signal_group = ['DC','DC_AE','DC_table','DC_Vib','table','all','AC','AC_AE','AC_table','AC_Vib','internals','ACDC']
    
class NASA_Dataset(Dataset):

    signal_list = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
    BASE_PROC_VAR_LIST = ['DOC', 'feed', 'material', 'Vc'] # Material is one of 1 (cast iron) and 2 (stainless steel)
    proc_variable_list = BASE_PROC_VAR_LIST.copy()
    
    TIME_DOMAIN_FEATURES = ['mean', 'var', 'rms', 'kurtosis', 'skewness', 'ptp']

    def __init__(self, 
                 transformX=None, 
                 transformProc=None, 
                 split='train',
                 split_type='runs',
                 split_ratios=(0.7, 0.15, 0.15), 
                 seed=42, 
                 signal_group='all', 
                 debug_plots=False, 
                 sliding_window_size=250,
                 sliding_window_stride=25,
                 runs_per_case_test_val=1,
                 split_offset=0,
                 use_generative=False,
                 generative_model_name=None,
                 gen_samples_per_combo=200,
                 extract_features=False,
                 apply_averaging=True,
                 avg_window_size=10):
        """
        Custom DataLoader for the NASA-Ames face-milling dataset with train/val/test split support.

        Args:
            transformX (torch.nn.Module or callable, optional): A transformation function or object applied 
                to the signal input data. For example, it could normalize the signals, augment data, or apply 
                domain-specific preprocessing. Defaults to None (no transformation).
            transformProc (torch.nn.Module or callable, optional): A transformation function or object applied 
                to the process input data. For example, it could normalize the signals, augment data, or apply 
                domain-specific preprocessing. Defaults to None (no transformation).
            split (str): Specifies the dataset split to load. Must be one of "train", "val", or "test".
                The split determines which preprocessed file is loaded.
            split_type (str): Specifies the dataset splitting method. Must be one of "runs" or "cases".
            split_ratios (tuple of floats): Proportions used to divide the full dataset into train, 
                validation, and test sets. The sum of the ratios should be 1. Defaults to (0.7, 0.15, 0.15).
            seed (int): Seed used for random number generation to ensure reproducible data splitting 
                and any other stochastic operations. Defaults to 42.
            signal_group (str): Name of the signal group used during training. This controls which signal 
                types are included in the dataset. Must be one of the `allowed_signal_group` values.
            debug_plots (bool): If True, enables debugging features like printing the min/max 
                statistics of each signal and generating plots for visualization. Defaults to False.
            sliding_window_size (int): Size of the sliding window (for data augmentation). Defaults to 250.
            sliding_window_stride (int): Stride of the sliding window. Defaults to 25.

        Attributes:
            self.data (torch.Tensor): Tensor containing the processed input signals.
            self.proc_data (torch.Tensor): Tensor containing additional processing parameters associated 
                with the input data.
            self.targets (torch.Tensor): Tensor containing the target labels for the dataset.

        Raises:
            AssertionError: If `split` is not one of "train", "val", or "test".
            AssertionError: If `split_type` is not one of "runs" or "cases".
            AssertionError: If `signal_group` is not in the allowed signal groups.

        Notes:
            - The NASA dataset files are expected to be preprocessed and stored in specific locations.
            If the files do not exist, the `_process_dataset()` method is called to generate and save 
            the splits.
            - If `debug_plots=True`, signal statistics will be printed to assist with dataset inspection 
            and debugging.
        """
        self.transformX = transformX
        self.transformProc = transformProc
        self.split = split
        self.seed = seed
        self.debug_plots = debug_plots
        self.signal_group = signal_group
        self.use_generative = use_generative
        self.generative_model_name = generative_model_name
        self.gen_samples_per_combo = gen_samples_per_combo
        self.avg_window_size = avg_window_size

        assert split in ['train','val','test'], f'Variable must be one of "train", "val", or "test"'
        assert split_type in ['runs', 'cases'], f'Variable must be one of "runs" or "cases"'
        assert signal_group in allowed_signal_group, f'Variable must be one of {allowed_signal_group}'

        data_file_name = f'data/nasa_{sliding_window_size}_{sliding_window_stride}{f"_ma{avg_window_size}" if apply_averaging else ""}.bin'

        if not os.path.exists(data_file_name):
            print('================== NASA dataset files do not exist\nProcessing and saving splits')
            self._process_dataset(data_file_name=data_file_name,apply_averaging=apply_averaging, avg_window_size=avg_window_size)
        else:
            print('================== NASA dataset files exist\nLoading pre-saved splits')

        # Load dat file with the data
        [x, x_process_params, y] = load(data_file_name)
        print(f'Samples: {x.shape}')
        
        total_data = len(x)

        #if split == 'train':
        #    if split_ratios[1] != 0:
        #        x, _, x_process_params, _, y, _ = train_test_split(xtr, xtr_proc, ytr, test_size=(split_ratios[1] / (split_ratios[0]+split_ratios[2])), random_state=self.seed)
        #    else: # When there is no validation, split, return the whole train set
        #        x, x_process_params, y = xtr, xtr_proc, ytr
        #elif split == 'val':
        #    _, x, _, x_process_params, _, y = train_test_split(xtr, xtr_proc, ytr, test_size=(split_ratios[1] / (split_ratios[0]+split_ratios[2])), random_state=self.seed)
        #else:
        #    x, x_process_params, y = xte, xte_proc, yte
        
        if split_type == 'runs':
            # Split runs per-case, then join. Ensure at least 1 run per case goes to the test set
            cases = np.unique(x_process_params[:, 0])
            train_pairs = []
            val_pairs = []
            test_pairs = []

            test_ratio = split_ratios[2]
            val_ratio = split_ratios[1]
            train_ratio = split_ratios[0]

            for c in np.sort(cases):
                # get unique runs for this case
                runs = np.unique(x_process_params[x_process_params[:, 0] == c, 1]).astype(int)
                n = len(runs)
                if n == 0:
                    continue
                
                # Random shuffle based on seed
                rng = np.random.RandomState(self.seed + int(c))
                perm = rng.permutation(n)
                runs_shuffled = runs[perm]
                
                if runs_per_case_test_val > 0:
                    # New logic: Cyclic selection based on offset
                    # Sort runs (already sorted by np.unique)
                    # Rotate runs by split_offset
                    effective_offset = split_offset % n
                    runs_ordered = np.concatenate((runs_shuffled[effective_offset:], runs_shuffled[:effective_offset]))
                    
                    # Select Test
                    test_sel = runs_ordered[:runs_per_case_test_val]
                    
                    # Select Val
                    val_sel = runs_ordered[runs_per_case_test_val : 2 * runs_per_case_test_val]
                    
                    # Select Train (rest)
                    tr = runs_ordered[2 * runs_per_case_test_val :]
                    vl = val_sel
                    
                else:
                    # Old logic: 
                    # deterministic shuffle per-case

                    # select test runs, ensure at least 1 run per case for test when test_ratio>0
                    if test_ratio > 0:
                        test_count = max(1, int(np.round(test_ratio * n)))
                        # limit test_count to at most n (if it becomes > n)
                        test_count = min(test_count, n)
                    else:
                        test_count = 0
                    test_sel = runs_shuffled[:test_count]

                    # remaining runs to be split into train/val
                    remaining = runs_shuffled[test_count:]
                    if val_ratio > 0 and len(remaining) > 1:
                        rel_val = val_ratio / (train_ratio + val_ratio)
                        tr, vl = train_test_split(
                            remaining, test_size=rel_val, random_state=self.seed
                        )
                    else:
                        # not enough runs to create a per-case validation split -> assign all to train
                        tr = remaining
                        vl = np.array([], dtype=int)

                # collect (case, run) pairs
                for r in tr:
                    train_pairs.append([int(c), int(r)])
                for r in vl:
                    val_pairs.append([int(c), int(r)])
                for r in test_sel:
                    test_pairs.append([int(c), int(r)])

            # convert to arrays (may be empty)
            self.train_runs = np.array(train_pairs, dtype=x_process_params[:, :2].dtype).reshape((-1, 2)) if len(train_pairs) > 0 else np.empty((0, 2))
            self.val_runs = np.array(val_pairs, dtype=x_process_params[:, :2].dtype).reshape((-1, 2)) if len(val_pairs) > 0 else np.empty((0, 2))
            self.test_runs = np.array(test_pairs, dtype=x_process_params[:, :2].dtype).reshape((-1, 2)) if len(test_pairs) > 0 else np.empty((0, 2))

            # Assign based on split
            if split == 'train':
                selected_runs = self.train_runs
            elif split == 'val':
                selected_runs = self.val_runs
            else:
                selected_runs = self.test_runs
        
            # Create boolean mask selecting all windows from selected runs
            if selected_runs.size == 0:
                mask = np.zeros(len(x_process_params), dtype=bool)
            else:
                mask = np.any((x_process_params[:, :2, None] == selected_runs.T).all(axis=1), axis=1)

            x, x_process_params, y = x[mask], x_process_params[mask], y[mask]
        
        elif split_type == 'cases':
        
            self.train_runs = [1, 2, 3, 4, 5, 6, 7, 8, 10, 14, 9, 13]
            self.val_runs   = [12, 16]
            self.test_runs  = [11, 15]
        
            if split == 'train':
                mask = np.isin(x_process_params[:, 0], self.train_runs)
            elif split == 'val':
                mask = np.isin(x_process_params[:, 0], self.val_runs)
            else:
                mask = np.isin(x_process_params[:, 0], self.test_runs)
        
            x, x_process_params, y = x[mask], x_process_params[mask], y[mask]
        x_process_params = x_process_params[:, 2:]
        print(f'================== Data splitted for {split} split: {len(x)/total_data*100}%')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')
        
        if debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)

        x, x_process_params, y = augment_signal(x, x_process_params, y, window_size=sliding_window_size, stride=sliding_window_stride)
        
        if debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)
                         
        print(f'================== Data augmented')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')

        if self.use_generative and self.generative_model_name and split == 'train':
            print(f'================== Generative Model Enabled ({self.generative_model_name}): Generating synthetic data')
            x_synth, x_proc_synth, y_synth = self._generate_synthetic_data(x, x_process_params, y, 
                                                                           sliding_window_size=sliding_window_size, 
                                                                           sliding_window_stride=sliding_window_stride)
             
            # Append synthetic data
            if len(x_synth) > 0:
                x = np.vstack([x, x_synth])
                x_process_params = np.vstack([x_process_params, x_proc_synth])
                y = np.concatenate([y, y_synth])
                print(f'Shapes after CVAE -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')
            else:
                 print('================== CVAE generated no data (check path or logic)')

        # Compute min and max for each column
        if debug_plots:
            x_stat = np.asarray(x, dtype=float)
            for i, signal in enumerate(NASA_Dataset.signal_list):
                print(f'Signal {signal}: Min = {np.min(x_stat[:,:,i])}, Max = {np.max(x_stat[:,:,i])}')
        
        if extract_features:
            print('================== Extracting time-domain features')
            
            # x is (N, W, C)
            # We want features (N, C*6)
            
            N, W, C = x.shape
            num_feats = len(NASA_Dataset.TIME_DOMAIN_FEATURES)
            
            # Pre-allocate
            x_features = np.zeros((N, C * num_feats), dtype=np.float32)
            
            # Order: For each channel, [mean, var, rms, kurt, skew, ptp]
            # features for channel i are at indices [i*6 : (i+1)*6]
            
            for c_idx in range(C):
                sig_data = x[:, :, c_idx] # (N, W)
                
                # Mean
                x_features[:, c_idx * num_feats + 0] = np.mean(sig_data, axis=1)
                # Variance
                x_features[:, c_idx * num_feats + 1] = np.var(sig_data, axis=1)
                # RMS
                x_features[:, c_idx * num_feats + 2] = np.sqrt(np.mean(sig_data**2, axis=1))
                # Kurtosis
                x_features[:, c_idx * num_feats + 3] = np.nan_to_num(kurtosis(sig_data, axis=1))
                # Skewness
                x_features[:, c_idx * num_feats + 4] = np.nan_to_num(skew(sig_data, axis=1))
                # Peak-to-Peak
                x_features[:, c_idx * num_feats + 5] = np.ptp(sig_data, axis=1)
                
            # Replace any remaining NaNs (just in case)
            x_features = np.nan_to_num(x_features)

            # Update proc_variable_list names
            new_proc_names = []
            for sig in NASA_Dataset.signal_list:
                for feat in NASA_Dataset.TIME_DOMAIN_FEATURES:
                    new_proc_names.append(f"{sig}_{feat}")
                    
            # Only update the class variable if it hasn't been updated yet (to avoid duplicates if re-instantiated)
            # But commonly we want instance-specific, yet the consumer uses the class static list.
            # Given the legacy design using class variable, we update it carefully or use instance var.
            # The user requested "update the proc_variable_list".
            NASA_Dataset.proc_variable_list = NASA_Dataset.BASE_PROC_VAR_LIST + new_proc_names

            # Concatenate features to process params
            # x_process_params is (N, 4)
            x_process_params = np.hstack([x_process_params, x_features])

        x, x_process_params = self.remove_signals(x, x_process_params)

        self.data = torch.Tensor(x).to(torch.float)
        self.proc_data = torch.Tensor(x_process_params).to(torch.float)
        self.targets =  torch.Tensor(y).unsqueeze(1).to(torch.float)

        print(f'================== NASA {split} dataset loaded for "{signal_group}"')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        sample = self.data[idx]
        sampleProc = self.proc_data[idx]

        if self.transformX is not None:
            sample = self.transformX(sample).to(dtype=torch.float)  # Apply the transform for the signal data

        if self.transformProc is not None:
            sampleProc = self.transformProc(sampleProc).to(dtype=torch.float)  # Apply the transform for the signal data

        sample = {
            'proc_data':sampleProc,
            'x':sample,
            }

        return sample, self.targets[idx]

    def _process_dataset(self,
                        apply_averaging = True,
                        avg_window_size=10, 
                        data_file_name='nasa.bin'):
        """
        Processes the dataset CSV file. If the expanded CSV file has not been created,
        the mat_to_csv.py script should be run to convert the dataset MAT file into an expanded
        csv file.
        
        Args:
            apply_averaging (bool): Apply moving average to denoise the signals.
            avg_window_size (int): Size of the moving average window.
            sliding_window_size (int): Size of the sliding window (for data augmentation).
            sliding_window_stride (int): Stride of the sliding window.
            data_file_name (str): Name of the file to save the data.
        """
        csv_file='data/mill_expanded.csv'
        if not os.path.exists(csv_file):
            mtc.convert_expanded(csv_file=csv_file)
        data_file = pd.read_csv(csv_file, sep=';', decimal='.', index_col=0).reset_index(drop=True)
        print('================== NASA dataset loaded from CSV')

        #Deleted cases
        #case 1 16, 17 (VB reduced)
        #case 2-5 (AE_T, noise)
        #case 2-6 (AE_S, noise)
        #case 7-4 (AE_T, noise)
        #case 8-3 (AE_T)
        #case 12-1 - Corrupt data and noise
        #case 12-12 - Singularity event
        data_file = data_file[(data_file['case'] != 1) | (data_file['run'] <= 15)]
        data_file = data_file[(data_file['case'] != 2) | (data_file['run'] != 5)]
        data_file = data_file[(data_file['case'] != 2) | (data_file['run'] != 6)]
        data_file = data_file[(data_file['case'] != 7) | (data_file['run'] != 4)]
        data_file = data_file[(data_file['case'] != 8) | (data_file['run'] != 3)]
        data_file = data_file[(data_file['case'] != 12) | (data_file['run'] != 1)]
        data_file = data_file[(data_file['case'] != 12) | (data_file['run'] != 12)]
        
        # Drop data for tool wear higher than 0.4 - No industrial applicability
        data_file = data_file[(data_file['VB'] <= 0.4)]

        data_file['Vc'] = 200
        data_file['Vc'] = data_file['Vc'].astype(float)
        data_file['VB interpolated'] = None
        data_file['VB interpolated'] = data_file['VB interpolated'].astype(float)
        for x in np.sort(data_file['case'].unique()):
            data_vb = data_file[(data_file['case'] == x)].groupby(['case','run']).mean().reset_index()
            data_vb = data_vb.interpolate()
            for y in np.sort(data_vb['run'].unique()):
                vb = data_vb.loc[data_vb['run']==y, 'VB'].iloc[0]
                data_file.loc[(data_file['case']==x) & (data_file['run']==y), 'VB interpolated'] = vb

        data_file = data_file.drop(columns=['time','VB']).rename(columns={'VB interpolated':'VB'})
        print('================== NASA dataset cleaned and VB interpolated')
        print(f'VB - min: {data_file["VB"].min()} - max: {data_file["VB"].max()}')

        # for signal in NASA_Dataset.signal_list:
        #     plt.plot(data_file[(data_file['case'] == 15) & (data_file['run'] == 4)][signal])
        # plt.show()

        x_proc_params = data_file[['case','run']+NASA_Dataset.BASE_PROC_VAR_LIST].copy().drop_duplicates()
        x_process_params = np.array(x_proc_params)
        for proc_var in NASA_Dataset.BASE_PROC_VAR_LIST:
            print(f'{proc_var} - min: {x_proc_params[proc_var].min()} - max: {x_proc_params[proc_var].max()}')

        y = data_file[['case','run','VB']].copy().drop_duplicates()
        y = np.array(y.drop(columns=['case','run'])).reshape((-1,))

        print('================== Transposing signals')
        x = NASA_Dataset.transpose_signals(data_file)
        print(f'Shape of signals after transposing:{x.shape}')
        print('================== Dataset splited into X, X_proc and Y')

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        if apply_averaging:
            print('================== Denoising signals with moving average')
            x = apply_moving_average(x, window_size=avg_window_size)
            x = np.array(x)
            print(f'Shape of signals after denoising:{x.shape}')
            print('================== Signals denoised')
        else:
            print('================== Signals were not desnoised')

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        #print('================== Augmenting signals')
        #x, x_process_params, y = augment_signal(x, x_process_params, y, window_size=sliding_window_size, stride=sliding_window_stride)

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list)

        x = np.transpose(x, (0, 2, 1))
        #print('Signals shape after augmentation:', x.shape)
        #print('Parameters shape after augmentation:', x_process_params.shape)
        #print('Labels shape after augmentation:', y.shape)
        #print('================== Data augmented')

        if self.debug_plots:
            plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)
        
        dump([x, x_process_params, y], data_file_name)

    def get_signal_list(self):
        signals = NASA_Dataset.signal_list.copy()
        if self.signal_group == 'DC':
            signals = ['smcDC']
        elif self.signal_group == 'DC_AE':
            signals = ['smcDC','AE_table']
        elif self.signal_group == 'DC_Vib':
            signals = ['smcDC','vib_table']
        elif self.signal_group == 'DC_table':
            signals = ['smcDC', 'vib_table','AE_table']
        elif self.signal_group == 'AC':
            signals = ['smcAC']
        elif self.signal_group == 'AC_AE':
            signals = ['smcAC','AE_table']
        elif self.signal_group == 'AC_Vib':
            signals = ['smcAC','vib_table']
        elif self.signal_group == 'AC_table':
            signals = ['smcAC', 'vib_table','AE_table']
        elif self.signal_group == 'table':
            signals = ['vib_table', 'AE_table']
        elif self.signal_group == 'ACDC':
            signals = ['smcAC', 'smcDC']
        elif self.signal_group == 'all':
            signals = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
        return signals

    def remove_signals(self, data, proc_data=None):
        idx = []
        signals = self.get_signal_list()
        
        # Find indices of signals to remove
        idx = [index for index, element in enumerate(NASA_Dataset.signal_list) if element not in signals]
        
        # Remove from signal data
        x = np.delete(data, idx, axis=2)
        
        if proc_data is not None:
            # We also need to remove the corresponding features from proc_data
            # proc_data structure: [DOC, feed, material, Vc] + [Sig1_Feats...] + [Sig2_Feats...] ...
            # Base vars count
            base_count = len(NASA_Dataset.BASE_PROC_VAR_LIST)
            feats_per_sig = len(NASA_Dataset.TIME_DOMAIN_FEATURES)
            
            # Identify columns to drop
            # For each removed signal index 'i' (in 0..5), we drop feature columns:
            # base_count + i*6 ... base_count + (i+1)*6 - 1
            
            cols_to_drop = []
            for i in idx:
                start_col = base_count + i * feats_per_sig
                end_col = start_col + feats_per_sig
                cols_to_drop.extend(range(start_col, end_col))
                
            proc_data = np.delete(proc_data, cols_to_drop, axis=1)
            
            # Update proc_variable_list to reflect removal
            # We reconstruct it based on kept signals
            current_proc_vars = NASA_Dataset.BASE_PROC_VAR_LIST.copy()
            for sig in signals: # These are the kept signals
                for feat in NASA_Dataset.TIME_DOMAIN_FEATURES:
                    current_proc_vars.append(f"{sig}_{feat}")
            
            NASA_Dataset.proc_variable_list = current_proc_vars
            
            return x, proc_data
            
        return x
    
    def transpose_signals(data):
        x = []

        for case in tqdm(np.sort(data['case'].unique())):
            data_case = data[(data['case'] == case)]
            for run in np.sort(data_case['run'].unique()):
                data_run = data_case[(data_case['run'] == run)]

                signals = []
                for col in NASA_Dataset.signal_list:
                    signals.append([])
                
                for _, row in data_run.iterrows():
                    for i, col in enumerate(NASA_Dataset.signal_list):
                        signals[i].append(row[col])
                x.append(signals)

        return np.array(x)

    def _generate_synthetic_data(self, x, x_process_params, y, sliding_window_size, sliding_window_stride):
        # 1. Map model name to tsgm class
        model_map = {
            'ConditionalVAE': tsgm.models.cBetaVAE,
            'TimeGAN': tsgm.models.timeGAN.TimeGAN
        }
        
        if self.generative_model_name not in model_map:
            print(f"Unknown generative model: {self.generative_model_name}")
            return [], [], []
            
        model_class = model_map[self.generative_model_name]
        output_dir = 'generative_results'
        suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
        
        model_file = os.path.join(output_dir, f"{self.generative_model_name}_{suffix}.weights.h5")
        scaler_file = os.path.join(output_dir, f"{self.generative_model_name}_scaler_{suffix}.joblib")
        
        # 2. Load Scalers
        if not os.path.exists(scaler_file):
             print(f"Generative Scaler file not found: {scaler_file}")
             return [], [], []
             
        stats = load(scaler_file)
        
        sig_mean = stats['signal_mean']
        sig_std = stats['signal_std']
        proc_min = stats['proc_min']
        proc_max = stats['proc_max']
        target_min = stats['target_min']
        target_max = stats['target_max']
        
        # 3. Load Model
        N, W, C = x.shape
        latent_dim = 16 
        
        
        # Build model with tsgm zoo architectures
        if self.generative_model_name == 'ConditionalVAE':
            architecture = tsgm.models.architectures.zoo["cvae_conv5"](
                seq_len=W, feat_dim=C, latent_dim=latent_dim, output_dim=5
            )
            model = tsgm.models.cvae.cBetaVAE(
                encoder=architecture.encoder, decoder=architecture.decoder, 
                latent_dim=latent_dim, temporal=False, beta=0.5
            )
        elif self.generative_model_name == 'TimeGAN':
            architecture = tsgm.models.architectures.zoo["cgan_lstm_3"](
                seq_len=W, feat_dim=C, latent_dim=24, output_dim=5
            )
            model = tsgm.models.cgan.ConditionalGAN(
                discriminator=architecture.discriminator, generator=architecture.generator,
                latent_dim=24, temporal=False
            )
        else:
            print(f"Unknown model: {self.generative_model_name}")
            return [], [], []
            
        if os.path.exists(model_file):
            model.load_weights(model_file)
        else:
             print(f"Generative Model file not found: {model_file}")
             return [], [], []
             
        # 4. Identify unique process parameter combinations
        base_params = x_process_params[:, :4] 
        unique_proc_combos = np.unique(base_params, axis=0)
        
        print(f"Generative: Found {len(unique_proc_combos)} unique process parameter combinations.")
        
        x_synth_list = []
        y_synth_list = []
        x_proc_synth_list = []
        
        # 5. Generate
        for combo in unique_proc_combos:
             targets = np.random.uniform(0, 0.4, size=(self.gen_samples_per_combo, 1))
             combo_repeated = np.tile(combo, (self.gen_samples_per_combo, 1))
             
             # Normalize combo
             proc_range = proc_max - proc_min
             proc_range[proc_range == 0] = 1.0
             combo_norm = (combo_repeated - proc_min) / proc_range
             
             # Normalize target
             target_range = target_max - target_min
             target_range[target_range == 0] = 1.0
             target_norm = (targets - target_min) / target_range
             
             condition = np.hstack([combo_norm, target_norm])
             
             # Generate using tsgm model API
             if self.generative_model_name == 'ConditionalVAE':
                  # Use cBetaVAE's generate method: generate(labels) -> (data, labels)
                  x_gen_norm, _ = model.generate(condition)
                  if hasattr(x_gen_norm, 'numpy'):
                       x_gen_norm = x_gen_norm.numpy()
             elif self.generative_model_name == 'TimeGAN':
                  # Use ConditionalGAN's generate method: generate(num, labels) -> data
                  x_gen_norm = model.generate(self.gen_samples_per_combo, condition)
                  if hasattr(x_gen_norm, 'numpy'):
                       x_gen_norm = x_gen_norm.numpy()
             
             # Convert to numpy if it's a tensor
             if hasattr(x_gen_norm, 'numpy'):
                  x_gen_norm = x_gen_norm.numpy()
                  
             # Denormalize Signal
             x_gen = x_gen_norm * (sig_std + 1e-8) + sig_mean
             
             x_synth_list.append(x_gen)
             y_synth_list.append(targets.flatten())
             x_proc_synth_list.append(combo_repeated)

        if len(x_synth_list) == 0:
             return [], [], []
             
        x_synth = np.vstack(x_synth_list)
        y_synth = np.concatenate(y_synth_list)
        x_proc_synth = np.vstack(x_proc_synth_list)
        
        return x_synth, x_proc_synth, y_synth

class MU_TCM_Dataset(Dataset):

    material={
        'CastIron.GG30':1,
        'StainlessSteel.316L':2,
        '':0
    }

    signals_250=['CV3_S', 'CV3_X', 'CV3_Y', 'CV3_Z', 'FREAL', 'POS_S', 'POS_X', 'POS_Y', 'POS_Z', 'SREAL', 'TV2_S', 'TV2_X', 'TV2_Y', 'TV2_Z', 'TV50', 'TV51']
    signals_50k=['Ax', 'Ay', 'Az', 'Fx', 'Fy', 'Fz']
    signals_1M=['AE_F', 'AE_RMS']
    
    data_file_name = 'data/mu.bin'

    signal_list = ['AE_F', 'AE_RMS', 'Ax', 'Ay', 'Az', 'CV3_S', 'CV3_X', 'CV3_Y', 'CV3_Z', 'FREAL', 'Fx', 'Fy', 'Fz', 'POS_S', 'POS_X', 'POS_Y', 'POS_Z', 'SREAL', 'TV2_S', 'TV2_X', 'TV2_Y', 'TV2_Z', 'TV50', 'TV51']
    proc_variable_list = ['ap', 'fz', 'material', 'Vc'] # Material is one of 1 (cast iron) and 2 (stainless steel)

    # Must end with a backslash (/)
    mu_data_folder = 'C:/MGEP Dropbox/Jose Joaquin Peralta Abadia/Tesis Joaquin Peralta/03_Proyectos/Captaciones/Dataset/'

    def __init__(self, 
                 transformX=None, 
                 transformProc=None, 
                 split='train', 
                 split_type='random',
                 split_ratios=(0.7, 0.15, 0.15), 
                 seed=42, 
                 signal_group='all', 
                 debug_plots=False, 
                 sliding_window_size=250,
                 sliding_window_stride=25,
                 filters=None,
                 use_synth=False,
                 synth_ratio=1):
        """
        Custom DataLoader for the MU-TCM face-milling dataset with train/val/test split support.
        
        Args:
            transformX (torch.nn.Module or callable, optional): A transformation function or object applied 
                to the signal input data. For example, it could normalize the signals, augment data, or apply 
                domain-specific preprocessing. Defaults to None (no transformation).
            transformProc (torch.nn.Module or callable, optional): A transformation function or object applied 
                to the process input data. For example, it could normalize the signals, augment data, or apply 
                domain-specific preprocessing. Defaults to None (no transformation).
            split (str): Specifies the dataset split to load. Must be one of "train", "val", or "test".
                The split determines which preprocessed file is loaded.
            split_ratios (tuple of floats): Proportions used to divide the full dataset into train, 
                validation, and test sets. The sum of the ratios should be 1. Defaults to (0.7, 0.15, 0.15).
            seed (int): Seed used for random number generation to ensure reproducible data splitting 
                and any other stochastic operations. Defaults to 42.
            signal_group (str): Name of the signal group used during training. This controls which signal 
                types are included in the dataset. Must be one of the `allowed_signal_group` values.
            debug_plots (bool): If True, enables debugging features like printing the min/max 
                statistics of each signal and generating plots for visualization. Defaults to False.
            sliding_window_size (int): Size of the sliding window (for data augmentation). Defaults to 250.
            sliding_window_stride (int): Stride of the sliding window. Defaults to 25.
            filters (dictionary): Columns to filter the data.

        Attributes:
            self.data (torch.Tensor): Tensor containing the processed input signals.
            self.proc_data (torch.Tensor): Tensor containing additional processing parameters associated 
                with the input data.
            self.targets (torch.Tensor): Tensor containing the target labels for the dataset.

        Raises:
            AssertionError: If `split` is not one of "train", "val", or "test".
            AssertionError: If `signal_group` is not in the allowed signal groups.

        Notes:
            - The MU-TCM dataset files are expected to be preprocessed and stored in specific locations.
            If the files do not exist, the `_process_dataset()` method is called to generate and save 
            the splits.
            - If `debug_plots=True`, signal statistics will be printed to assist with dataset inspection 
            and debugging.
        """
        self.transformX = transformX
        self.transformProc = transformProc
        self.split = split
        self.seed = seed
        self.debug_plots = debug_plots
        self.signal_group = signal_group
        
        assert split in ['train','val','test'], f'Variable must be one of "train", "val", or "test"'
        assert signal_group in allowed_signal_group, f'Variable must be one of {allowed_signal_group}'

        data_file_name = f'data/mu_{sliding_window_size}_{sliding_window_stride}.bin'

        if not os.path.exists(data_file_name):
            print('================== MU-TCM dataset files do not exist\nProcessing and saving splits...')
            self._process_dataset(sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride, data_file_name=data_file_name)
        else:
            print('================== MU-TCM dataset files exist\nLoading pre-saved splits...')

        # Load dat file with the data
        [x, x_process_params, y] = load(data_file_name)
        print(f'Samples: {len(x)}')

        if filters is not None:
            for key, value in filters.items():
                col_idx = MU_TCM_Dataset.proc_variable_list.index(key)
                idx = np.asarray(x_process_params[:,col_idx]==value).nonzero()[0]
                x = x[idx]
                x_process_params = x_process_params[idx]
                y = y[idx]
        
        xtr, xte, xtr_proc, xte_proc, ytr, yte = train_test_split(x, x_process_params, y, test_size=split_ratios[2], random_state=self.seed)
        
        total_data = len(x)
        
        # Extract unique runs as groups (case-run pairs)
        unique_runs = np.unique(x_process_params, axis=0)
    
        # Split runs (not windows)
        self.train_runs, self.test_runs = train_test_split(
            unique_runs,
            test_size=split_ratios[2],
            random_state=self.seed
        )
                
        if split_ratios[1] != 0:
            self.train_runs, self.val_runs = train_test_split(
                self.train_runs,
                test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]),
                random_state=self.seed
            )
        else:
            self.val_runs = np.empty((0, 2))  # no validation
    
        # Assign based on split
        if split == 'train':
            selected_runs = self.train_runs
        elif split == 'val':
            selected_runs = self.val_runs
        else:
            selected_runs = self.test_runs
    
        # Create boolean mask selecting all windows from selected runs
        mask = np.any(
            (x_process_params[:, :2, None] == selected_runs.T).all(axis=1), axis=1
        )
    
        x, x_process_params, y = x[mask], x_process_params[mask], y[mask]

        print(f'================== Data splitted for {split} split: {len(x)/total_data*100}%')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')
        
        x, x_process_params, y = augment_signal(x, x_process_params, y, window_size=sliding_window_size, stride=sliding_window_stride)
        print(f'================== Data augmented')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')

        if use_synth:
            print(f'Generating synthetic data with a ratio of 1:{synth_ratio}')

            print('=== Min and max values for real data')
            print(f'Process data: Min: {np.min(x_process_params, axis=0)} - Max: {np.max(x_process_params, axis=0)}')
            print(f'VB: Min: {np.min(y)} - Max: {np.max(y)}')

            filter_string = str(filters['material']) if 'material' in filters.keys() else 'X'
            filter_string += f"_{filters['fz'] if 'fz' in filters.keys() else 'X'}"
            filter_string += f"_{filters['Vc'] if 'Vc' in filters.keys() else 'X'}"

            synth_data_file_name = f'data/mu_{sliding_window_size}_{sliding_window_stride}_synth{synth_ratio}_{filter_string}.bin'
            if os.path.exists(synth_data_file_name):
                [synthetic_x, synthetic_proc, synthetic_y] = load(synth_data_file_name)
            else:
                synthetic_x, synthetic_proc, synthetic_y = augment_dataset_softdtw(
                    x, x_process_params, y, num_synthetic_ts=len(x)*synth_ratio, max_neighbors=5
                )            
                dump([synthetic_x, synthetic_proc, synthetic_y], synth_data_file_name)

            x = np.vstack([x, synthetic_x])
            y = np.hstack([y, synthetic_y])
            x_process_params = np.vstack([x_process_params, synthetic_proc])

            print('=== Min and max values for synthetic data')
            print(f'Process data: Min: {np.min(x_process_params, axis=0)} - Max: {np.max(x_process_params, axis=0)}')
            print(f'VB: Min: {np.min(y)} - Max: {np.max(y)}')

            print(f'Shapes after augmenting -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')

        # Compute min and max for each column
        if debug_plots:
            x_stat = np.asarray(x, dtype=float)
            for i, signal in enumerate(MU_TCM_Dataset.signal_list):
                print(f'Signal {signal}: Min = {np.min(x_stat[:,:,i])}, Max = {np.max(x_stat[:,:,i])}')
        
        x = self.remove_signals(x)

        self.data = torch.Tensor(x).to(torch.float)
        self.proc_data = torch.Tensor(x_process_params).to(torch.float)
        self.targets =  torch.Tensor(y).unsqueeze(1).to(torch.float)
        print('================== MU-TCM dataset loaded')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    
        sample = self.data[idx]
        sampleProc = self.proc_data[idx]

        if self.transformX is not None:
            sample = self.transformX(sample).to(dtype=torch.float)  # Apply the transform for the signal data

        if self.transformProc is not None:
            sampleProc = self.transformProc(sampleProc).to(dtype=torch.float)  # Apply the transform for the signal data

        sample = {
            'proc_data':sampleProc,
            'x':sample,
            }

        return sample, self.targets[idx]

    def _process_dataset(self,
                        apply_averaging = True,
                        avg_window_size=20, 
                        sliding_window_size=250,
                        sliding_window_stride=25,
                        data_file_name='mu.bin'):
        """
        Processes the dataset MAT files.
        
        Args:
            apply_averaging (bool): Apply moving average to denoise the signals.
            avg_window_size (int): Size of the moving average window.
            sliding_window_size (int): Size of the sliding window (for data augmentation).
            sliding_window_stride (int): Stride of the sliding window.
            debug_plots (boolean): Plot signals for debugging of dataset processing.
        """
        if not os.path.exists(MU_TCM_Dataset.mu_data_folder):
            print(f"Error: The MU-TCM dataset directory '{MU_TCM_Dataset.mu_data_folder}' does not exist.")
            return
        files = get_file_list(root=MU_TCM_Dataset.mu_data_folder+'signals_synced', pattern='*.mat')
        print(f'File quantity: {len(files)}')

        # Load the signal stats CSV file
        csv_file = MU_TCM_Dataset.mu_data_folder+'signals_stats.csv'  # Replace with the path to your CSV file
        df = pd.read_csv(csv_file,sep=';',decimal='.')

        # Convert the DataFrame to a dictionary
        stats_dict = df.set_index('_file_name').to_dict(orient='index')
        
        x = []
        x_proc_params = []
        y = []
        loaded_files = []

        for i, file in enumerate(tqdm(files)):
            mat = MU_TCM_Dataset.load_mat_file(file)
            process_variables = []
            signals = {}
            for proc_var in MU_TCM_Dataset.proc_variable_list:
                if proc_var == 'material':
                    process_variables.append(MU_TCM_Dataset.material[mat['WorkpieceMaterial']])
                else:
                    process_variables.append(mat[proc_var])
            for signal in MU_TCM_Dataset.signal_list:
                if signal in mat.keys():
                    signals[signal] = mat[signal]
                elif self.signal_group in('DC'):
                    signals[signal] = np.zeros(0)
            if len(signals) == len(MU_TCM_Dataset.signal_list):
                x.append(signals)
                x_proc_params.append(process_variables)
                y.append(mat['VB'])
                loaded_files.append(file)
        print('================== MU-TCM dataset loaded from MAT files')

        x_proc_params = np.asarray(x_proc_params, dtype=float)
        y = np.asarray(y, dtype=float)
        
        print(f'Shape of signals after loading: ({len(x)}, {len(x[0])}, x)')

        print(f'VB min: {min(y)} - VB max: {max(y)}')
        for i, proc_var in enumerate(MU_TCM_Dataset.proc_variable_list):
            print(f'{proc_var} - min: {x_proc_params[:,i].min()} - max: {x_proc_params[:,i].max()}')

        if self.debug_plots:
            plot_signals_dict(x, MU_TCM_Dataset.signal_list)

        if len(MU_TCM_Dataset.signals_50k) > 0 or (len(MU_TCM_Dataset.signals_1M) > 0):
            print('================== Downsampling high-frequency signals')
            for i in tqdm(range(len(x))):
                signals = []
                for signal in MU_TCM_Dataset.signal_list:
                    if (signal in MU_TCM_Dataset.signals_1M or signal in MU_TCM_Dataset.signals_50k) and len(x[i][signal]) > 0:
                        x[i][signal] = downsample(x[i][signal], len(x[i]['SREAL']))
                    if len(x[i][signal]) > 0:
                        signals.append(x[i][signal])
                    else:
                        signals.append(np.zeros(len(x[i]['SREAL'])))
                x[i] = np.array(signals)
            print('================== Dataset downsampled and splited into X, X_proc and Y')
        else:
            print('================== Dataset splited into X, X_proc and Y')

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list)

        print('================== Removing exit and entry portions of the signal')
        for i, file in enumerate(tqdm(loaded_files)):
            file = ntpath.basename(file)
            signal_start, signal_end = stats_dict[file]['SREAL_start'], stats_dict[file]['SREAL_end']
            x[i] = x[i][:, signal_start:signal_end]
        print('================== Exit and entry portions of the signal removed')

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list)

        if apply_averaging:
            print('================== Denoising signals with moving average')
            x = apply_moving_average(x, window_size=avg_window_size)
            print('================== Signals denoised')
        else:
            print('================== Signals were not desnoissed')

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list)

        # print('================== Augmenting signals')
        # x, x_proc_params, y = augment_signal(x, x_proc_params, y, window_size=sliding_window_size, stride=sliding_window_stride)

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list)

        x = np.transpose(x, (0, 2, 1))
        # print('Signals shape after augmentation:', x.shape)
        # print('Parameters shape after augmentation:', x_proc_params.shape)
        # print('Labels shape after augmentation:', y.shape)
        # print('================== Data augmented')

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list, signal_chanel=2)

        dump([x, x_proc_params, y], data_file_name)

    def get_signal_list(self):
        signals = NASA_Dataset.signal_list.copy()
        if self.signal_group in ('DC','AC'):
            signals = ['CV3_S']
        elif self.signal_group in ('DC_AE','AC_AE'):
            signals = ['CV3_S','AE_F']
        elif self.signal_group in ('DC_Vib','AC_Vib'):
            signals = ['CV3_S','Ay']
        elif self.signal_group in ('DC_table','AC_table'):
            signals = ['CV3_S','Ay','AE_F']
        elif self.signal_group in ('table'):
            signals = ['Ay','AE_F']
        elif self.signal_group in ('internals'):
            signals = ['CV3_S', 'CV3_X', 'CV3_Y', 'CV3_Z', 'FREAL', 'SREAL', 'TV2_S', 'TV2_X', 'TV2_Y', 'TV2_Z', 'TV50', 'TV51']
        return signals

    def remove_signals(self, data):
        idx = []
        signals = self.get_signal_list()
        idx = [index for index, element in enumerate(MU_TCM_Dataset.signal_list) if element not in signals]
        x = np.delete(data, idx, axis=2)
        return x

    def load_mat_file(file):
        mat = scipy.io.loadmat(file)
        for k in mat.keys():
            if type(mat[k]) == np.ndarray:
                if len(mat[k].shape) == 1:
                    mat[k] = str(mat[k][0])
                elif mat[k].shape[0] == 1 and mat[k].shape[1]>1:
                    mat[k] = mat[k][0]
                elif mat[k].shape[0] == 1:
                    mat[k] = mat[k][0][0]
                else:
                    mat[k] = mat[k].reshape((len(mat[k])))
        return mat
    
    
__all__ = [
    "MU_TCM_Dataset",
    "NASA_Dataset",
]