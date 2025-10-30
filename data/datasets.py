import os
import ntpath
import numpy as np
import pandas as pd
from joblib import load, dump
from tqdm import tqdm

import scipy.io
from sklearn.model_selection import train_test_split

from helpers import plot_signals, plot_signals_dict, augment_signal, apply_moving_average, downsample, get_file_list, augment_dataset_softdtw, compute_min_max
import data.mat_to_csv as mtc

import torch
from torch.utils.data import Dataset

allowed_signal_group = ['DC','DC_AE','DC_table','DC_Vib','table','all','AC','AC_AE','AC_table','AC_Vib','internals','ACDC']
    
class NASA_Dataset(Dataset):

    signal_list = ['smcAC', 'smcDC', 'vib_table', 'vib_spindle', 'AE_table', 'AE_spindle']
    proc_variable_list = ['DOC', 'feed', 'material', 'Vc'] # Material is one of 1 (cast iron) and 2 (stainless steel)

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
                 sliding_window_stride=25):
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
            scalogram_cwt (str): The wavelet to use to generate scalograms, e.g., 'morse'. Defaults to None. 

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

        assert split in ['train','val','test'], f'Variable must be one of "train", "val", or "test"'
        assert split_type in ['runs', 'cases'], f'Variable must be one of "runs" or "cases"'
        assert signal_group in allowed_signal_group, f'Variable must be one of {allowed_signal_group}'

        data_file_name = f'data/nasa_{sliding_window_size}_{sliding_window_stride}.bin'

        if not os.path.exists(data_file_name):
            print('================== NASA dataset files do not exist\nProcessing and saving splits')
            self._process_dataset(sliding_window_size=sliding_window_size, sliding_window_stride=sliding_window_stride, data_file_name=data_file_name)
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
            # Extract unique runs as groups (case-run pairs)
            unique_runs = np.unique(x_process_params[:, :2], axis=0)
        
            # Split runs (not windows)
            train_runs, test_runs = train_test_split(
                unique_runs,
                test_size=split_ratios[2],
                random_state=self.seed
            )
                    
            if split_ratios[1] != 0:
                train_runs, val_runs = train_test_split(
                    train_runs,
                    test_size=split_ratios[1] / (split_ratios[0] + split_ratios[1]),
                    random_state=self.seed
                )
            else:
                val_runs = np.empty((0, 2))  # no validation
            print("Train runs", train_runs.T)
            print("Val runs", val_runs.T)
            print("Test runs", test_runs.T)
        
            # Assign based on split
            if split == 'train':
                selected_runs = train_runs
            elif split == 'val':
                selected_runs = val_runs
            else:
                selected_runs = test_runs
        
            # Create boolean mask selecting all windows from selected runs
            mask = np.any(
                (x_process_params[:, :2, None] == selected_runs.T).all(axis=1), axis=1
            )
        
            x, x_process_params, y = x[mask], x_process_params[mask], y[mask]
        
        elif split_type == 'cases':
        
            train_cases = [1, 2, 3, 4, 5, 6, 7, 8]
            val_cases   = [9, 10, 13, 14]
            test_cases  = [11, 12, 15, 16]
        
            if split == 'train':
                mask = np.isin(x_process_params[:, 0], train_cases)
            elif split == 'val':
                mask = np.isin(x_process_params[:, 0], val_cases)
            else:
                mask = np.isin(x_process_params[:, 0], test_cases)
        
            x, x_process_params, y = x[mask], x_process_params[mask], y[mask]
        x_process_params = x_process_params[:, 2:]
        print(f'================== Data splitted for {split} split: {len(x)/total_data*100}%')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')
        
        x, x_process_params, y = augment_signal(x, x_process_params, y, window_size=sliding_window_size, stride=sliding_window_stride)
        print(f'================== Data augmented')
        print(f'Shapes -> X: {x.shape} - X process params: {x_process_params.shape} - Y: {y.shape}')

        # Compute min and max for each column
        if debug_plots:
            x_stat = np.asarray(x, dtype=float)
            for i, signal in enumerate(NASA_Dataset.signal_list):
                print(f'Signal {signal}: Min = {np.min(x_stat[:,:,i])}, Max = {np.max(x_stat[:,:,i])}')
        
        x = self.remove_signals(x)

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
                        avg_window_size=20, 
                        sliding_window_size=250,
                        sliding_window_stride=25,
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

        x_proc_params = data_file[['case','run']+NASA_Dataset.proc_variable_list].copy().drop_duplicates()
        x_process_params = np.array(x_proc_params)
        for proc_var in NASA_Dataset.proc_variable_list:
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

        #if self.debug_plots:
        #    plot_signals(x, NASA_Dataset.signal_list, signal_chanel=2)
        
        dump([x, x_process_params, y], data_file_name)

    def remove_signals(self, data):
        idx = []
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
        idx = [index for index, element in enumerate(NASA_Dataset.signal_list) if element not in signals]
        x = np.delete(data, idx, axis=2)
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
        if split_type == 'random':
            if split == 'train':
                if split_ratios[1] != 0:
                    x, _, x_process_params, _, y, _ = train_test_split(xtr, xtr_proc, ytr, test_size=(split_ratios[1] / (split_ratios[0]+split_ratios[2])), random_state=self.seed)
                else: # When there is no validation, split, return the whole train set
                    x, x_process_params, y = xtr, xtr_proc, ytr
            elif split == 'val':
                _, x, _, x_process_params, _, y = train_test_split(xtr, xtr_proc, ytr, test_size=(split_ratios[1] / (split_ratios[0]+split_ratios[2])), random_state=self.seed)
            else:
                x, x_process_params, y = xte, xte_proc, yte

        print(f'================== Data splitted for {split} split: {len(x)/total_data*100}%')
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

        print('================== Augmenting signals')
        x, x_proc_params, y = augment_signal(x, x_proc_params, y, window_size=sliding_window_size, stride=sliding_window_stride)

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list)

        x = np.transpose(x, (0, 2, 1))
        print('Signals shape after augmentation:', x.shape)
        print('Parameters shape after augmentation:', x_proc_params.shape)
        print('Labels shape after augmentation:', y.shape)
        print('================== Data augmented')

        if self.debug_plots:
            plot_signals(x, MU_TCM_Dataset.signal_list, signal_chanel=2)

        dump([x, x_proc_params, y], data_file_name)

    def remove_signals(self, data):
        idx = []
        signals = MU_TCM_Dataset.signal_list.copy()
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