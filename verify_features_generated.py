import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from data.datasets import NASA_Dataset
except ImportError:
    print("Could not import NASA_Dataset. Make sure you run this from the project root.")
    sys.exit(1)

def verify_features():
    print("Initializing NASA_Dataset (train split)...")
    # minimal parameters to avoid heavy processing if possible, or just standard
    try:
        ds = NASA_Dataset(split='train', signal_group='all', sliding_window_size=250, sliding_window_stride=25)
    except Exception as e:
        print(f"Failed to initialize dataset: {e}")
        return

    x_sample, y_sample = ds[0]
    # x_sample is a dict {'proc_data': ..., 'x': ...}
    
    proc_data = x_sample['proc_data']
    signal_data = x_sample['x']
    
    print(f"Signal Data Shape: {signal_data.shape}")
    print(f"Proc Data Shape: {proc_data.shape}")
    
    # Expected process variables: 4 base + 6 signals * 6 features = 40
    expected_proc_dim = 4 + 6 * 6
    if proc_data.shape[0] == expected_proc_dim:
        print(f"SUCCESS: Proc data dimension is {proc_data.shape} as expected.")
        print(f'The values of the first sample are: {proc_data}')
    else:
        print(f"FAILURE: Expected {expected_proc_dim} proc features, got {proc_data.shape[0]}.")

    print("\nChecking proc_variable_list...")
    print(f"List length: {len(NASA_Dataset.proc_variable_list)}")
    print(f"List content: {NASA_Dataset.proc_variable_list}")
    
    if len(NASA_Dataset.proc_variable_list) == expected_proc_dim:
        print("SUCCESS: proc_variable_list length matches.")
    else:
        print("FAILURE: proc_variable_list length mismatch.")

    # Test removing signals
    print("\nTesting remove_signals...")
    # remove_signals is called internally based on signal_group, let's try a different group (hacky via new instance)
    try:
        ds_dc = NASA_Dataset(split='train', signal_group='DC', sliding_window_size=250, sliding_window_stride=25)
        # DC group has 1 signal: smcDC
        # Expected features: 4 + 1 * 6 = 10
        x_dc, y_dc = ds_dc[0]
        proc_dc = x_dc['proc_data']
        print(f"DC Group Proc Data Shape: {proc_dc.shape}")
        
        if proc_dc.shape[0] == 10:
            print("SUCCESS: Signal removal correctly reduced feature space.")
            print(f'The values of the first sample are: {proc_dc}')
        else:
            print(f"FAILURE: Expected 10 features for DC group, got {proc_dc.shape[0]}.")
             
    except Exception as e:
        print(f"Failed during remove_signals test: {e}")

if __name__ == "__main__":
    verify_features()
