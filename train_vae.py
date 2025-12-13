
import os
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from data.datasets import NASA_Dataset
from models.dl_models import create_cvae
from helpers import compute_mean_std
from data.transforms import StdScalerTransform
from joblib import dump
import matplotlib.pyplot as plt

def train_cvae(sliding_window_size=250, sliding_window_stride=25, output_dir='vae_results', epochs=200):
    # Parameters
    latent_dim = 16
    batch_size = 32
    beta = 1.0 # KL weight
    learning_rate = 1e-3
    
    # Generate path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Loading Dataset...")
    # Load dataset (using 'train' split for learning)
    
    dataset = NASA_Dataset(split='train', signal_group='all', 
                           sliding_window_size=sliding_window_size, 
                           sliding_window_stride=sliding_window_stride)
    
    # Normalize Data
    mean, std = compute_mean_std(dataset)
    scaler = StdScalerTransform(mean=mean, std=std)
    
    # Extract data from dataset
    x_train = dataset.data # Tensor (N, W, C)
    x_proc = dataset.proc_data # Tensor (N, P_total)
    y_train = dataset.targets # Tensor (N, 1)
    
    # Identify Base Process Params (first 4 columns)
    # ['DOC', 'feed', 'material', 'Vc']
    base_proc_params = x_proc[:, :4]
    
    # Normalize Process Params (MinMax)
    # We compute min/max from data
    proc_min = tf.reduce_min(base_proc_params, axis=0)
    proc_max = tf.reduce_max(base_proc_params, axis=0)
    
    # Avoid division by zero if min==max (constant param)
    proc_range = proc_max - proc_min
    proc_range = tf.where(proc_range == 0, tf.ones_like(proc_range), proc_range)
    
    base_proc_norm = (base_proc_params - proc_min) / proc_range
    
    # Normalize Target (MinMax)
    # Target VB is 0 to ~max.
    y_min = tf.reduce_min(y_train, axis=0)
    y_max = tf.reduce_max(y_train, axis=0)
    y_range = y_max - y_min
    y_range = tf.where(y_range == 0, tf.ones_like(y_range), y_range)
    
    y_train_norm = (y_train - y_min) / y_range
    
    # Condition Vector: Concat Proc + Target
    # shape: (N, 4 + 1) = (N, 5)
    condition_vector = tf.concat([base_proc_norm, y_train_norm], axis=1)
    
    # Normalize Signal Data
    x_train_norm = scaler(x_train).numpy()
    condition_vector_np = condition_vector.numpy()
    
    feature_dim = x_train_norm.shape[2]
    seq_len = x_train_norm.shape[1]
    label_dim = condition_vector_np.shape[1]
    
    print(f"Data Shape: {x_train_norm.shape}")
    print(f"Condition Vector Shape: {condition_vector_np.shape}")
    
    # Create CVAE
    cvae = create_cvae(input_shape=(seq_len, feature_dim), 
                       label_dim=label_dim, 
                       latent_dim=latent_dim,
                       filters=[32, 64, 128], 
                       kernel_size=[3, 3, 3], 
                       strides=[2, 2, 2])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    cvae.compile(optimizer=optimizer)
    
    # Train
    print("Starting Training...")
    
    # Fit expects x=input, y=target. 
    # For custom train_step in CVAE, we pass (x, condition) as 'x' in the tuple logic?
    # Our CVAE.train_step handles `data`.
    # If we call fit(x=signal, y=condition), `data` in train_step will be (signal, condition).
    
    # Callbacks
    es = EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=1)
    
    cvae.fit(x_train_norm, condition_vector_np, epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[es])
    
    cvae.save_weights(os.path.join(output_dir, 'cvae_weightsv2.h5'))
    print("Training Complete. Weights saved.")
    
    # Save Scaler Stats for Dataset usage
    # We need: mean(signal), std(signal), min(proc), max(proc), min(target), max(target)
    scaler_stats = {
        'signal_mean': mean,
        'signal_std': std,
        'proc_min': proc_min.numpy(),
        'proc_max': proc_max.numpy(),
        'target_min': y_min.numpy(),
        'target_max': y_max.numpy()
    }
    dump(scaler_stats, os.path.join(output_dir, 'vae_scaler_stats.joblib'))
    print("Scaler stats saved.")
    
    # # Generation
    # print("Generating Synthetic Data...")
    
    # # Define generation targets (0 to 0.4 mm)
    # # We can generate uniformly distributed targets or sample from real distribution.
    # # Let's generate 1000 samples uniformly distributed.
    # num_samples = 1000
    # random_targets = np.random.uniform(0, 0.4, size=(num_samples, 1))
    
    # # Sample latent vectors from prior N(0, I)
    # z_sample = np.random.normal(size=(num_samples, latent_dim))
    # print(f'Size of z_sample: {z_sample.shape}')
    
    # # Decode
    # decoder = cvae.decoder
    # synthetic_x_norm = decoder.predict([z_sample, random_targets])
    
    # # Denormalize (Inverse Transform)
    # # Scaler: y = (x - mean) / std  => x = y * std + mean
    # # Ensure mean/std are numpy
    # mean_np = mean.numpy() if hasattr(mean, 'numpy') else mean
    # std_np = std.numpy() if hasattr(std, 'numpy') else std
    
    # synthetic_x = synthetic_x_norm * (std_np + 1e-8) + mean_np
    
    # # Generate "Process Params" for synthetic data?
    # # Usually we just need (X, Proc, Y).
    # # Dataset expects proc_data. 
    # # For now, we can fill proc_data with plausible values if needed, or just 0s.
    # # Since proc_data includes extracted features, we should ideally re-compute features from synthetic X?
    # # Yes, if we want consistency. But NASA_Dataset computes features on load/init.
    # # If we save this as a .bin file compatible with load(), we need [X, X_proc, Y].
    # # But wait, NASA_Dataset expects X as raw signals and then Augments them.
    # # Here we are generating Windows directly (augmented/windowed view).
    # # If we want to use this data, we probably need a dedicated dataset class or bypass the windowing logic.
    # # OR, we save it as "augmented" data and modify dataset loader to accept pre-augmented synthetic data.
    # # Actually, NASA_Dataset has `synth_ratio` logic in `_process_dataset`? 
    # # No, that was in `MU_TCM_Dataset`. `NASA_Dataset` doesn't seem to have `augment_dataset_softdtw` integrated in init/loading for synthetic data injection yet.
    # # For this task, "propose implementation" -> I will save the result to a file `cvae_generated.bin`.
    
    # # Process Params: We don't have them for synthetic data. We can calculate the features (mean/var etc) from generated signals.
    # # But the "base" process params (DOC, feed, etc) are unknown unless we condition on them too.
    # # For now, let's create placeholders for base params, and compute time-domain features.
    
    # # Let's just save X and Y for inspection.
    # save_path = os.path.join(output_dir, 'cvae_generated.bin')
    # dump([synthetic_x, random_targets], save_path)
    # print(f"Synthetic data saved to {save_path}")
    
    # # Plot some examples
    # plt.figure(figsize=(10, 4))
    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     # Plot first channel
    #     plt.plot(synthetic_x[i, :, 0])
    #     plt.title(f"Target: {random_targets[i][0]:.2f}")
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, 'synthetic_examples.png'))
    # print("Examples plotted.")

    # print("Examples plotted.")
    
def main():
    train_cvae()

if __name__ == "__main__":
    main()
