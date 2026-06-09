
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def interactive_mat_to_csv(mat_file='data/mill.mat', csv_file='data/mill_interactive.csv'):
    """Interactively review each experiment in the NASA Ames mill.mat file and export to CSV.

    For each experiment the raw signals are plotted. The operator can adjust
    the start and end sample indices to trim sensor warm-up and run-out, then
    confirm the cuts before the excerpt is appended to the output CSV.

    Args:
        mat_file (str): Path to the source ``mill.mat`` MATLAB file.
            Defaults to ``'data/mill.mat'``.
        csv_file (str): Destination CSV path for the exported excerpts.
            Defaults to ``'data/mill_interactive.csv'``.
    """
    print(f"Loading {mat_file}...")
    try:
        mat = scipy.io.loadmat(mat_file)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return
    
    mat = {k: v for k, v in mat.items() if k[0] != '_'}
    
    measurements = {}
    data_info = {}
    
    # Defaults for cuts (can be adjusted per experiment)
    current_start = 2000
    current_end = 7200
    
    signal_names_display = ["smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"]

    for key, value in mat.items():
        experiments_array = value[0] 
        total_experiments = len(experiments_array)
        
        print(f"\nFound {total_experiments} experiments under key '{key}'.")
        
        for i in range(total_experiments):
            sub_arr = experiments_array[i]
            
            scalars = []
            signals = {}
            
            for sub_index in range(len(sub_arr)):
                item = sub_arr[sub_index]
                
                if len(item) == 1:
                    val = item[0][0]
                    scalars.append(val)
                else:
                    signals[sub_index] = item.flatten()
            
            print(f"\n========================================")
            print(f"Experiment {i+1} / {total_experiments}")
            print(f"Process Info: {scalars}")
            print(f"========================================")
            
            confirmed = False            
            sorted_signal_indices = sorted(signals.keys())
            
            while not confirmed:
                print("Plotting raw signals... (Close the plot window to proceed)")
                _, axes = plt.subplots(len(sorted_signal_indices), 1, figsize=(10, 15), sharex=True)
                if len(sorted_signal_indices) == 1:
                    axes = [axes]
                
                for idx, sig_idx in enumerate(sorted_signal_indices):
                    ax = axes[idx]
                    ax.plot(signals[sig_idx])
                    
                    ax.axvline(x=current_start, color='g', linestyle='--', linewidth=2, label='Start')
                    if current_end is not None:
                        ax.axvline(x=current_end, color='r', linestyle='--', linewidth=2, label='End')
                    
                    label_name = signal_names_display[idx] if idx < len(signal_names_display) else f"Signal {idx}"
                    ax.set_ylabel(label_name)
                    ax.grid(True)
                
                axes[-1].set_xlabel("Time (Samples)")
                plt.suptitle(f"Experiment {i+1} - Raw Data")
                plt.tight_layout()
                plt.show()
                
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
                
                print("Plotting with proposed cuts... (Close the plot window to confirm)")
                _, axes = plt.subplots(len(sorted_signal_indices), 1, figsize=(10, 15), sharex=True)
                if len(sorted_signal_indices) == 1:
                    axes = [axes]
                
                for idx, sig_idx in enumerate(sorted_signal_indices):
                    ax = axes[idx]
                    ax.plot(signals[sig_idx])
                    
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
                
                conf = input("Are these cuts correct? (y/n) [y]: ").strip().lower()
                if conf in ['', 'y', 'yes']:
                    confirmed = True
                else:
                    print("re-starting selection for this experiment...")

            data_info[i] = scalars
            
            if i not in measurements:
                measurements[i] = {}
            
            for sig_idx in sorted_signal_indices:
                original_signal = sub_arr[sig_idx]
                sliced_signal = original_signal[current_start:current_end]
                measurements[i][sig_idx] = sliced_signal

    print("\nAll experiments processed. Building CSV...")
    
    counter = 0
    data_expanded = {}
    
    for k in sorted(data_info.keys()):
        first_sig_idx = list(measurements[k].keys())[0]
        signal_len = len(measurements[k][first_sig_idx])
        
        for t in range(signal_len):
            row = []            
            row.extend(data_info[k])
            
            for sig_idx in sorted(measurements[k].keys()):
                val = measurements[k][sig_idx][t][0]
                row.append(val)
                
            data_expanded[counter] = row
            counter += 1
            
    print("Creating DataFrame...")
    cols = ['case','run','VB', 'time','DOC',"feed", "material", 
            "smcAC","smcDC","vib_table","vib_spindle","AE_table","AE_spindle"]
    
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
