
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def interactive_mat_to_csv(mat_file='data/mill.mat', csv_file='data/mill_interactive.csv'):
    print(f"Loading {mat_file}...")
    try:
        mat = scipy.io.loadmat(mat_file)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    # Clean metadata (remove keys starting with __)
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    
    # We assume the main data is in the first key (e.g., 'mill')
    # If there are multiple keys, this loop handles them as per original logic (though effectively overwriting if indices collide)
    
    measurements = {} # Structure: {experiment_index: {signal_index: signal_array}}
    data_info = {}    # Structure: {experiment_index: [scalar_values]}
    
    # Initialize Defaults
    # Original default was 2000 to 7200
    current_start = 2000
    current_end = 7200
    
    # Hardcoded signal names for display (matching the expected output columns)
    signal_names_display = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

    for key, value in mat.items():
        # param 'value' is expected to be a structured array, e.g., (1, 167)
        experiments_array = value[0] 
        total_experiments = len(experiments_array)
        
        print(f"\nFound {total_experiments} experiments under key '{key}'.")
        
        for i in range(total_experiments):
            sub_arr = experiments_array[i] # This is the data for one experiment
            
            scalars = []
            signals = {} # Temporary holder for signals found in this experiment: index -> array
            
            # Iterate through the items in the experiment structure
            # sub_arr is an array of arrays/values
            for sub_index in range(len(sub_arr)):
                item = sub_arr[sub_index]
                
                # Check based on shape/length if it's a scalar or signal
                # Original logic: len(item) == 1 -> scalar (vals = item[0][0])
                # Note: item is usually shape (1,1) for scalars in loaded mat files, or (N, 1) for signals
                
                if len(item) == 1:
                    # Scalar / Info
                    val = item[0][0]
                    scalars.append(val)
                else:
                    # Signal
                    # Flattenting to 1D for easier plotting
                    signals[sub_index] = item.flatten()
            
            # Show Info
            print(f"\n========================================")
            print(f"Experiment {i+1} / {total_experiments}")
            print(f"Process Info: {scalars}")
            print(f"========================================")
            
            # Interactive Loop
            confirmed = False
            
            sorted_signal_indices = sorted(signals.keys())
            
            while not confirmed:
                # 1. Plot Raw
                print("Plotting raw signals... (Close the plot window to proceed)")
                fig, axes = plt.subplots(len(sorted_signal_indices), 1, figsize=(10, 15), sharex=True)
                if len(sorted_signal_indices) == 1:
                    axes = [axes]
                
                for idx, sig_idx in enumerate(sorted_signal_indices):
                    ax = axes[idx]
                    ax.plot(signals[sig_idx])
                    
                    # Draw Cut Lines
                    ax.axvline(x=current_start, color='g', linestyle='--', linewidth=2, label='Start')
                    if current_end is not None:
                        ax.axvline(x=current_end, color='r', linestyle='--', linewidth=2, label='End')
                    
                    # Try to label with meaningful name if index matches
                    label_name = signal_names_display[idx] if idx < len(signal_names_display) else f"Signal {idx}"
                    ax.set_ylabel(label_name)
                    ax.grid(True)
                
                axes[-1].set_xlabel("Time (Samples)")
                plt.suptitle(f"Experiment {i+1} - Raw Data")
                plt.tight_layout()
                plt.show() # Blocks until closed
                
                # 2. Ask for Indices
                print(f"Current Range: Start={current_start}, End={current_end}")
                s_in = input("Enter Start Index [Press Enter to keep current]: ").strip()
                if s_in:
                    try:
                        current_start = int(s_in)
                    except ValueError:
                        print("Invalid start index. Using previous.")
                
                e_in = input("Enter End Index [Press Enter to keep current]: ").strip()
                if e_in:
                    try:
                        current_end = int(e_in)
                    except ValueError:
                        print("Invalid end index. Using previous.")
                
                # 3. Plot with Cuts (Verification)
                print("Plotting with proposed cuts... (Close the plot window to confirm)")
                fig, axes = plt.subplots(len(sorted_signal_indices), 1, figsize=(10, 15), sharex=True)
                if len(sorted_signal_indices) == 1:
                    axes = [axes]
                
                for idx, sig_idx in enumerate(sorted_signal_indices):
                    ax = axes[idx]
                    ax.plot(signals[sig_idx])
                    
                    # Draw Cut Lines
                    ax.axvline(x=current_start, color='g', linestyle='--', linewidth=2, label='Start')
                    if current_end is not None:
                        ax.axvline(x=current_end, color='r', linestyle='--', linewidth=2, label='End')
                    
                    label_name = signal_names_display[idx] if idx < len(signal_names_display) else f"Signal {idx}"
                    ax.set_ylabel(label_name)
                    ax.grid(True)
                    if idx == 0:
                        ax.legend()

                axes[-1].set_xlabel("Time (Samples)")
                plt.suptitle(f"Experiment {i+1} - Verify Cut [{current_start} : {current_end}]")
                plt.tight_layout()
                plt.show()
                
                # 4. Confirmation
                conf = input("Are these cuts correct? (y/n) [y]: ").strip().lower()
                if conf in ['', 'y', 'yes']:
                    confirmed = True
                else:
                    print("re-starting selection for this experiment...")

            # Store the data
            # Store Info
            data_info[i] = scalars # equivalent to 'data[i] = lst'
            
            # Store Processed Signals
            if i not in measurements:
                measurements[i] = {}
            
            for sig_idx in sorted_signal_indices:
                original_signal = sub_arr[sig_idx] # Keep original shape (N, 1) usually
                # Slice
                # Handle cases where slice is out of bounds implicitly handled by python slicing usually, 
                # but let's be safe
                sliced_signal = original_signal[current_start:current_end]
                measurements[i][sig_idx] = sliced_signal

    print("\nAll experiments processed. Building CSV...")

    # Flatten logic from original script
    # Columns map: 
    # original used columns=['case','run','VB', 'time','DOC',"feed", "material", "smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"]
    # scalars map to first N columns.
    
    counter = 0
    data_expanded = {}
    
    # Iterate in order of keys
    for k in sorted(data_info.keys()):
        # info_vals = data_info[k]
        
        # Get length of signals (they should all be same length after slicing)
        # We grab the first available signal in measurements[k]
        first_sig_idx = list(measurements[k].keys())[0]
        signal_len = len(measurements[k][first_sig_idx])
        
        # Original: values_len = len(list(measurements[k].values())[0])
        
        # Logic: One row per time step
        for t in range(signal_len):
            row = []
            
            # 1. Add scalars (info)
            # data_info[k] is List of values
            # However, original script logic:
            # for sub_v in range(len(v)): if v[sub_v] is not None: lst.append(v[sub_v])
            # v is 'data' dict value which matches 'scalars' here.
            row.extend(data_info[k])
            
            # 2. Add signal values at time t
            # Original: for _, v2 in measurements[k].items(): lst.append(v2[i][0])
            # They iterate dict items. Py3.7+ preserves insertion order.
            # We sorted keys during capture? No, 'measurements' keys insertion order matters if we want to match columns.
            # In extraction loop, we iterated range(len(sub_arr)).
            # measurements[k] keys are indices.
            # We should iterate sorted keys to be deterministic.
            
            for sig_idx in sorted(measurements[k].keys()):
                # array shape is (N, 1) usually
                val = measurements[k][sig_idx][t][0]
                row.append(val)
                
            data_expanded[counter] = row
            counter += 1
            
    print("Creating DataFrame...")
    cols = ['case','run','VB', 'time','DOC',"feed", "material", 
            "smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"]
    
    # Check if column count matches row length
    # If not, we might warn, but we try to proceed.
    # User asked to replicate exactly.
    
    df = pd.DataFrame.from_dict(data_expanded, orient='index', columns=cols)
    
    print("Sorting values...")
    try:
        df = df.sort_values(['case','run','time'])
    except KeyError:
        print("Warning: Could not sort by case/run/time (columns might be missing or scalar values mismatch).")
    
    print(f"Saving to {csv_file}...")
    df.to_csv(csv_file, sep=';', decimal='.')
    print("Done!")

if __name__ == "__main__":
    interactive_mat_to_csv()
