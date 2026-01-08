import keras
from keras import layers
import keras.ops as ops
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
from joblib import dump, load
from helpers import compute_mean_std
from data.transforms import StdScalerTransform, MinMaxScalerTransform

import gc

# tsgm imports
import tsgm

from tqdm import tqdm
import numpy as np

import torch
import matplotlib.pyplot as plt


def train_generative_model(model_class_name, dataset, sliding_window_size, sliding_window_stride, 
                           latent_dim=16, epochs=200, batch_size=128, output_dir='generative_results', 
                           normalization_type='minmax'):
    """
    Generic training function for generative models using tsgm zoo architectures.
    """
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting training for {model_class_name}...")

    # Normalize Data
    x_train_raw = ops.convert_to_tensor(dataset.data) # Tensor (N, W, C)
    
    mean_tensor = None
    std_tensor = None
    data_min = None
    data_max = None
    scaler = None

    if normalization_type == 'minmax':
        # MinMax Normalization
        x_flat = ops.reshape(x_train_raw, (-1, x_train_raw.shape[2]))
        data_min = ops.min(x_flat, axis=0) # (C,)
        data_max = ops.max(x_flat, axis=0) # (C,)
        scaler = MinMaxScalerTransform(min=data_min, max=data_max)
    elif normalization_type == 'standard':
        # Standard (Z-score) Normalization
        mean, std = compute_mean_std(dataset)
        mean_tensor = ops.convert_to_tensor(mean)
        std_tensor = ops.convert_to_tensor(std)
        scaler = StdScalerTransform(mean=mean_tensor, std=std_tensor)
    else:
        raise ValueError(f"Unknown normalization_type: {normalization_type}")
    # Extract data from dataset
    x_train = ops.convert_to_tensor(dataset.data) # Tensor (N, W, C)
    x_proc = ops.convert_to_tensor(dataset.proc_data) # Tensor (N, P_total)
    y_train = ops.convert_to_tensor(dataset.targets) # Tensor (N, 1)
    
    # Condition: DOC, feed, material, Vc (first 4) + target (last)
    base_proc_params = x_proc[:, :4]
    
    # Normalize Process Params (MinMax)
    proc_min = ops.min(base_proc_params, axis=0)
    proc_max = ops.max(base_proc_params, axis=0)
    proc_range = proc_max - proc_min
    proc_range = ops.where(proc_range == 0, ops.ones_like(proc_range), proc_range)
    base_proc_norm = (base_proc_params - proc_min) / proc_range
    
    # Normalize Target (MinMax)
    y_min = ops.min(y_train, axis=0)
    y_max = ops.max(y_train, axis=0)
    y_range = y_max - y_min
    y_range = ops.where(y_range == 0, ops.ones_like(y_range), y_range)
    y_train_norm = (y_train - y_min) / y_range
    
    condition_vector = ops.concatenate([base_proc_norm, y_train_norm], axis=1)
    
    # Normalize Signal Data
    x_train_norm = scaler(x_train)
    
    feature_dim = x_train_norm.shape[2]
    seq_len = x_train_norm.shape[1]
    cond_dim = condition_vector.shape[1]
    
    # Save Scaler Stats and Params ONCE (Global)
    suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
    
    scaler_stats = {'normalization_type': normalization_type}
    if normalization_type == 'minmax':
        scaler_stats['signal_min'] = ops.convert_to_numpy(data_min)
        scaler_stats['signal_max'] = ops.convert_to_numpy(data_max)
    elif normalization_type == 'standard':
        scaler_stats['signal_mean'] = ops.convert_to_numpy(mean_tensor)
        scaler_stats['signal_std'] = ops.convert_to_numpy(std_tensor)
        
    scaler_stats.update({
        'proc_min': ops.convert_to_numpy(proc_min),
        'proc_max': ops.convert_to_numpy(proc_max),
        'target_min': ops.convert_to_numpy(y_min),
        'target_max': ops.convert_to_numpy(y_max)
    })
    stats_path = os.path.join(output_dir, f"{model_class_name}_scaler_{suffix}.joblib")
    dump(scaler_stats, stats_path)
    print(f"Scaler stats saved to {stats_path}")

    # Loop over channels to train separate models
    models = []
    
    for ch in range(feature_dim):
        print(f"Training {model_class_name} for Channel {ch}/{feature_dim-1}...")
        
        # Extract single channel data: (N, T, 1)
        x_train_ch = x_train_norm[:, :, ch:ch+1]
        
        # Use tsgm zoo architectures (feat_dim=1)
        if model_class_name == 'ConditionalVAE':
            # Use cvae_conv5 from zoo
            # https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/VAEs/cVAE.ipynb
            architecture = tsgm.models.architectures.zoo["cvae_conv5"](
                seq_len=seq_len, 
                feat_dim=1, # Single channel
                latent_dim=latent_dim, 
                output_dim=cond_dim
            )
            encoder = architecture.encoder
            decoder = architecture.decoder
            
            model = tsgm.models.cvae.cBetaVAE(
                encoder=encoder, 
                decoder=decoder, 
                latent_dim=latent_dim,
                temporal=False,  # Static labels per sample
                beta=0.5
            )
            optimizer = keras.optimizers.Adam(learning_rate=1e-3)
            model.compile(optimizer=optimizer)
            
        elif model_class_name == 'TimeGAN':
            # Use cgan_lstm_3 from zoo
            # https://github.com/AlexanderVNikitin/tsgm/blob/main/tutorials/GANs/cGAN.ipynb
            architecture = tsgm.models.architectures.zoo["cgan_lstm_3"](
                seq_len=seq_len, 
                feat_dim=1, # Single channel
                latent_dim=latent_dim, 
                output_dim=cond_dim
            )
            discriminator = architecture.discriminator
            generator = architecture.generator
           
            model = tsgm.models.cgan.ConditionalGAN(
                discriminator=discriminator,
                generator=generator,
                latent_dim=latent_dim,
                temporal=False  # Static labels per sample
            )
            d_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
            g_optimizer = keras.optimizers.Adam(learning_rate=1e-4)
            loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
            model.compile(d_optimizer=d_optimizer, g_optimizer=g_optimizer, loss_fn=loss_fn)
        else:
            raise ValueError(f"Unknown generative model name: {model_class_name}")
            
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5, min_delta=0.1, restore_best_weights=True)

        # Train
        print(f"Fitting model for Channel {ch}...")
        model.fit(x=x_train_ch, y=condition_vector, epochs=epochs, 
                  batch_size=batch_size)#, callbacks=[es])
        
        # --- Debug: Save Reconstruction Plot for first few channels ---
        # Select random sample
        idx = np.random.randint(0, x_train_ch.shape[0])
        sample_x = x_train_ch[idx:idx+1] # (1, T, 1)
        sample_c = condition_vector[idx:idx+1] # (1, C)
        print(f'Shape sample X {sample_x.shape} - Shape sample C {sample_c.shape}')
            
        recon = None
        if model_class_name == 'ConditionalVAE':
            # cBetaVAE call inputs: [x, cond]
            print('Making predictions with cVAE')
            recon = model.predict([sample_x, sample_c], verbose=0)
        print(f'Shape of reconstructions {recon.shape}')
        
        if recon is not None:
            # Generate new sample from same condition
            gen_sample = None
            if model_class_name == 'ConditionalVAE':
                    gen_sample, _ = model.generate(sample_c)
            elif model_class_name == 'TimeGAN':
                    gen_sample = model.generate(sample_c)

            plt.figure(figsize=(10, 4))
            plt.plot(sample_x[0, :, 0].cpu().numpy(), label='Original')
            plt.plot(recon[0, :, 0], label='Reconstruction')
            if gen_sample is not None:
                    # Move to cpu/numpy if needed
                    if hasattr(gen_sample, 'cpu'): gen_sample = gen_sample.cpu().detach().numpy()
                    plt.plot(gen_sample[0, :, 0], label='Generated', alpha=0.7, linestyle='--')
            
            plt.title(f'Train Debug: Ch{ch} Recon vs Gen (Epochs={epochs})')
            plt.legend()
            debug_path = os.path.join(output_dir, f'debug_recon_gen_{model_class_name}_ch{ch}.png')
            print(f'Debug path for plots is {debug_path}')
            plt.savefig(debug_path)
            plt.close()
            print(f"Saved debug reconstruction/generation plot to {debug_path}")
                
        # Save Model Weights for Channel
        
        # Save Model Weights for Channel
        weights_path = os.path.join(output_dir, f"{model_class_name}_{suffix}_ch{ch}.weights.h5")
        model.save_weights(weights_path)
        print(f"Channel {ch} weights saved to {weights_path}")
        
        models.append(model)
        
        # Cleanup Channel Model
        # del model
        # keras.utils.clear_session()
        # gc.collect()
        # torch.cuda.empty_cache()
    
    print(f"Training complete for all {feature_dim} channels.")
    
    # Cleanup
    keras.utils.clear_session()
    gc.collect()
    
    return models


def generate_synthetic_data(model_class_name, gen_config, num_samples, 
                            condition_data=None, x_train_info=None, latent_dim=16):
    """
    Generate synthetic data using a trained generative model.
    Replicates the logic used in PreprocessingTuner for consistency, including balanced sampling.
    """
    
    output_dir = gen_config['output_dir']
    sliding_window_size = gen_config['sliding_window_size']
    sliding_window_stride = gen_config['sliding_window_stride']
    curr_mean = gen_config['current_mean']
    curr_std = gen_config['current_std']
    normalization_type = gen_config['normalization_type']
    
    suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
    
    # Load Stats and Params
    stats_path = os.path.join(output_dir, f"{model_class_name}_scaler_{suffix}.joblib")
    if not os.path.exists(stats_path):
        print(f"Error: Scaler stats not found at {stats_path}. Aborting.")
        return None, None
        
    scaler_stats = load(stats_path)
    print(scaler_stats)
    # Prefer normalization type from saved stats, fallback to config
    gen_norm_type = scaler_stats.get('normalization_type', normalization_type)
    
    gen_min = None
    gen_max = None
    gen_mean = None
    gen_std = None

    if gen_norm_type == 'minmax':
        gen_min = scaler_stats['signal_min']
        gen_max = scaler_stats['signal_max']
        feature_dim = gen_min.shape[0]
    elif gen_norm_type == 'standard':
        gen_mean = scaler_stats['signal_mean']
        gen_std = scaler_stats['signal_std']
        feature_dim = gen_mean.shape[0]
    
    cond_dim = condition_data.shape[1]
    seq_len = sliding_window_size
    
    # Check if ANY channel weights exist to decide whether to proceed
    first_ch_path = os.path.join(output_dir, f"{model_class_name}_{suffix}_ch0.weights.h5")
    if not os.path.exists(first_ch_path):
         print(f"Warning: Channel 0 weights not found at {first_ch_path}. Skipping.")
         return None, None

    # Prepare Sampling Indices ONCE (Shared across all channels to maintain correlations if any, though independent generation breaks inter-channel corr)
    # Balanced Sampling Logic
    if x_train_info is not None:
        unique_proc, unique_indices = np.unique(x_train_info, axis=0, return_inverse=True)
        num_unique = len(unique_proc)
        n_per_unique = num_samples // num_unique
        remainder = num_samples % num_unique
        
        idx = []
        for i in range(num_unique):
            matching_indices = np.where(unique_indices == i)[0]
            count = n_per_unique + (1 if i < remainder else 0)
            if len(matching_indices) > 0:
                sampled = np.random.choice(matching_indices, count, replace=True)
                idx.extend(sampled)
        idx = np.array(idx)
        np.random.shuffle(idx)
        sampled_cond = condition_data[idx]
    else:
        # Fallback to simple random sampling
        idx = np.random.randint(0, condition_data.shape[0], num_samples)
        sampled_cond = condition_data[idx]
    
    generated_channels = []
    
    for ch in range(feature_dim):
        print(f"Generating {model_class_name} for Channel {ch}/{feature_dim-1}...")
        weights_path = os.path.join(output_dir, f"{model_class_name}_{suffix}_ch{ch}.weights.h5")
        
        if not os.path.exists(weights_path):
            print(f"Error: Weights for channel {ch} not found. Aborting.")
            return None, None
            
        # Instantiate Model Architecture (feat_dim=1)
        model = None
        if model_class_name == 'ConditionalVAE':
            architecture = tsgm.models.architectures.zoo["cvae_conv5"](
                seq_len=seq_len, feat_dim=1, latent_dim=latent_dim, output_dim=cond_dim
            )
            model = tsgm.models.cvae.cBetaVAE(
                encoder=architecture.encoder, decoder=architecture.decoder,
                latent_dim=latent_dim, temporal=False, beta=0.5
            )
        elif model_class_name == 'TimeGAN':
            architecture = tsgm.models.architectures.zoo["cgan_lstm_3"](
                seq_len=seq_len, feat_dim=1, latent_dim=latent_dim, output_dim=cond_dim
            )
            model = tsgm.models.cgan.ConditionalGAN(
                discriminator=architecture.discriminator, generator=architecture.generator,
                latent_dim=latent_dim, temporal=False
            )
        
        # Build & Load
        try:
            input_shapes = [(None, seq_len, 1), (None, cond_dim)]
            model.build(input_shapes)
            model.load_weights(weights_path)
        except Exception as e:
            print(f"Error loading generative model {model_class_name} Channel {ch}: {e}")
            return None, None
        
        # Generate Channel Data
        channel_signal = []
        batch_size = 100
        num_batches = int(np.ceil(num_samples / batch_size))
        
        for b in tqdm(range(num_batches), desc=f"Gen Ch {ch}"):
            start = b * batch_size
            end = min((b + 1) * batch_size, num_samples)
            batch_cond = sampled_cond[start:end]
            
            batch_syn = None
            if model_class_name == "ConditionalVAE":
                 batch_syn, _ = model.generate(batch_cond)
            elif model_class_name == "TimeGAN":
                 batch_syn = model.generate(batch_cond)
                 
            if batch_syn is not None:
                 if hasattr(batch_syn, 'numpy'):
                     batch_syn = batch_syn.cpu().detach().numpy()
                 elif hasattr(batch_syn, 'detach'):
                     batch_syn = batch_syn.detach().cpu().numpy()
                 channel_signal.append(batch_syn)
    
            del batch_syn
            torch.cuda.empty_cache()
            
        if len(channel_signal) > 0:
            channel_full = np.concatenate(channel_signal, axis=0) # (N, T, 1)
            generated_channels.append(channel_full)
        else:
             return None, None
             
        # Cleanup Channel Model
        del model
        keras.utils.clear_session()
        gc.collect()
        torch.cuda.empty_cache()

    # Concatenate Channels
    if len(generated_channels) == feature_dim:
        synthetic_signal = np.concatenate(generated_channels, axis=2) # (N, T, C)
    else:
        print("Error: Dimension mismatch in generated channels.")
        return None, None

        
    # Denormalize (Generator Scale) -> Normalize (Current Run Scale)
    print(f'Current mean: {curr_mean} - Std: {curr_std}')
    
    syn_raw = None
    if gen_norm_type == 'minmax':
        # Inverse MinMax: x_raw = x_norm * (max - min) + min
        gen_range = gen_max - gen_min
        gen_range = np.where(gen_range == 0, 1.0, gen_range)
        syn_raw = synthetic_signal * gen_range + gen_min
    elif gen_norm_type == 'standard':
        syn_raw = synthetic_signal * gen_std + gen_mean
        
    safe_std = np.where(curr_std == 0, 1.0, curr_std)
    syn_renorm = (syn_raw - curr_mean) / safe_std
    
    # Cleanup
    keras.utils.clear_session()
    gc.collect()
    torch.cuda.empty_cache()
    
    return syn_renorm, sampled_cond
