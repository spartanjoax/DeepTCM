import keras
from keras import layers
import keras.ops as ops
import numpy as np
import os
from joblib import dump
from helpers import compute_mean_std
from data.transforms import StdScalerTransform

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim))
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon

class ConditionalVAE(keras.Model):
    """
    Conditional Variational Autoencoder for signal generation.
    Using Conv1D/Conv1DTranspose for better signal feature extraction.
    Conditioned on process variables and target wear.
    """
    def __init__(self, input_shape, cond_dim, latent_dim=16, **kwargs):
        super().__init__(**kwargs)
        self.input_shape_val = input_shape # (Time, Channels)
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # --- Encoder ---
        encoder_inputs = layers.Input(shape=input_shape)
        cond_inputs = layers.Input(shape=(cond_dim,))
        
        x = layers.Conv1D(32, kernel_size=3, strides=2, padding="same", activation="relu")(encoder_inputs)
        x = layers.Conv1D(64, kernel_size=3, strides=2, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        
        # Concatenate condition at the bottleneck
        x_cond = layers.Concatenate()([x, cond_inputs])
        x_enc = layers.Dense(64, activation="relu")(x_cond)
        x_enc = layers.Dense(32, activation="relu")(x_enc)
        
        z_mean = layers.Dense(latent_dim, name="z_mean")(x_enc)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x_enc)
        z = Sampling()([z_mean, z_log_var])
        
        self.encoder = keras.Model([encoder_inputs, cond_inputs], [z_mean, z_log_var, z], name="encoder")

        # --- Decoder ---
        latent_inputs = layers.Input(shape=(latent_dim,))
        decoder_cond_inputs = layers.Input(shape=(cond_dim,))
        
        # Merge latent and condition
        x_dec = layers.Concatenate()([latent_inputs, decoder_cond_inputs])
        x_dec = layers.Dense(32, activation="relu")(x_dec)
        x_dec = layers.Dense(64, activation="relu")(x_dec)
        
        # Dynamic calculation for Conv1DTranspose start shape
        # We did 2 strides of 2 in encoder, so Time -> Time // 4
        # Note: This might need adjustment based on exact seq_len
        initial_time = input_shape[0] // 4
        x_dec = layers.Dense(initial_time * 64, activation="relu")(x_dec)
        x_dec = layers.Reshape((initial_time, 64))(x_dec)
        
        x_dec = layers.Conv1DTranspose(64, kernel_size=3, strides=2, padding="same", activation="relu")(x_dec)
        x_dec = layers.Conv1DTranspose(32, kernel_size=3, strides=2, padding="same", activation="relu")(x_dec)
        
        # Final layer to match input_shape (Time, Channels)
        # We use a final Conv1D and possibly a Cropping layer if there's a slight mismatch
        # For simplicity in this implementation, we assume Time is a multiple of 4.
        outputs = layers.Conv1D(input_shape[1], kernel_size=3, padding="same", activation="linear")(x_dec)
        
        self.decoder = keras.Model([latent_inputs, decoder_cond_inputs], outputs, name="decoder")

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # data is ([x, cond], x) - typical VAE training input/output
        if isinstance(data, tuple) or isinstance(data, list):
            x, cond = data[0]
        else:
            x, cond = data
            
        with ops.gradient_tape.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([x, cond])
            reconstruction = self.decoder([z, cond])
            
            reconstruction_loss = ops.mean(
                ops.sum(
                    keras.losses.mean_squared_error(x, reconstruction),
                    axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
            
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        x, cond = inputs
        _, _, z = self.encoder([x, cond])
        return self.decoder([z, cond])

class TimeGAN(keras.Model):
    """
    TimeGAN architecture for synthetic signal generation.
    Simplified Keras 3 implementation based on the original TimeGAN paper.
    """
    def __init__(self, seq_len, n_features, cond_dim, latent_dim=24, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_features = n_features
        self.cond_dim = cond_dim
        self.latent_dim = latent_dim

        # 1. Embedder
        e_in = layers.Input(shape=(seq_len, n_features))
        x = layers.GRU(latent_dim, return_sequences=True)(e_in)
        x = layers.GRU(latent_dim, return_sequences=True)(x)
        e_out = layers.Dense(latent_dim, activation="sigmoid")(x)
        self.embedder = keras.Model(e_in, e_out, name="embedder")

        # 2. Recovery
        r_in = layers.Input(shape=(seq_len, latent_dim))
        x = layers.GRU(latent_dim, return_sequences=True)(r_in)
        x = layers.GRU(latent_dim, return_sequences=True)(x)
        r_out = layers.Dense(n_features, activation="linear")(x)
        self.recovery = keras.Model(r_in, r_out, name="recovery")

        # 3. Generator
        g_in = layers.Input(shape=(seq_len, latent_dim))
        g_cond = layers.Input(shape=(cond_dim,))
        # Repeat condition for each time step
        repeated_cond = layers.RepeatVector(seq_len)(g_cond)
        x = layers.Concatenate()([g_in, repeated_cond])
        x = layers.GRU(latent_dim, return_sequences=True)(x)
        x = layers.GRU(latent_dim, return_sequences=True)(x)
        g_out = layers.Dense(latent_dim, activation="sigmoid")(x)
        self.generator = keras.Model([g_in, g_cond], g_out, name="generator")

        # 4. Discriminator
        d_in = layers.Input(shape=(seq_len, latent_dim))
        x = layers.GRU(latent_dim, return_sequences=True)(d_in)
        x = layers.GRU(latent_dim, return_sequences=True)(x)
        d_out = layers.Dense(1, activation="linear")(x) # Logits
        self.discriminator = keras.Model(d_in, d_out, name="discriminator")

    def call(self, inputs):
        # inputs: [noise, cond]
        return self.recovery(self.generator(inputs))

    # TimeGAN training logic involves multiple loss functions
    def train_step(self, data):
        # x: (Batch, SeqLen, Features), cond: (Batch, CondDim)
        x, cond = data[0]
        
        # 1. Embedding Network training (Supervised + Reconstruction)
        with ops.gradient_tape.GradientTape() as tape:
            h = self.embedder(x)
            x_tilde = self.recovery(h)
            h_hat_supervise = self.generator([h[:, :-1, :], cond]) # Supervised next-step
            
            # Reconstruction Loss
            loss_recovery = ops.mean(keras.losses.mean_squared_error(x, x_tilde))
            # Supervised Loss
            loss_supervised = ops.mean(keras.losses.mean_squared_error(h[:, 1:, :3], h_hat_supervise[:, :, :3]))
            
            loss_embed = loss_recovery + 0.1 * loss_supervised
            
        grads = tape.gradient(loss_embed, self.embedder.trainable_weights + self.recovery.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.embedder.trainable_weights + self.recovery.trainable_weights))

        # 2. Generator & Discriminator training (Adversarial)
        # (Simplified for Keras 3 train_step)
        batch_size = ops.shape(x)[0]
        z = keras.random.normal(shape=(batch_size, self.seq_len, self.latent_dim))
        
        with ops.gradient_tape.GradientTape() as gen_tape, ops.gradient_tape.GradientTape() as disc_tape:
            h_fake = self.generator([z, cond])
            x_fake = self.recovery(h_fake)
            
            y_real = self.discriminator(h)
            y_fake = self.discriminator(h_fake)
            
            # Adversarial Losses
            loss_disc = ops.mean(keras.losses.binary_crossentropy(ops.ones_like(y_real), ops.sigmoid(y_real))) + \
                        ops.mean(keras.losses.binary_crossentropy(ops.zeros_like(y_fake), ops.sigmoid(y_fake)))
            
            loss_gen = ops.mean(keras.losses.binary_crossentropy(ops.ones_like(y_fake), ops.sigmoid(y_fake)))
            
        # Optimization
        # In a real TimeGAN, these are carefully weighted.
        return {"loss_embed": loss_embed, "loss_gen": loss_gen, "loss_disc": loss_disc}

class TimeGrad(keras.Model):
    """
    TimeGrad implementation using Denoising Diffusion.
    Conditioned on process variables and wear.
    """
    def __init__(self, seq_len, n_features, cond_dim, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.n_features = n_features
        self.cond_dim = cond_dim
        
        # Denoising Backbone
        self.backbone = keras.Sequential([
            layers.Input(shape=(seq_len, n_features + cond_dim + 1)),
            layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            layers.LSTM(64, return_sequences=True),
            layers.Dense(n_features)
        ])
        
    def train_step(self, data):
        x, cond = data[0]
        batch_size = ops.shape(x)[0]
        t = keras.random.uniform(shape=(batch_size, 1, 1), minval=0, maxval=1)
        noise = keras.random.normal(shape=ops.shape(x))
        
        # Add noise based on t (Simplified Diffusion)
        x_noisy = x * (1 - t) + noise * t
        
        # Prepare inputs for backbone
        # Condition is (Batch, CondDim) -> Repeat to (Batch, SeqLen, CondDim)
        cond_rep = ops.repeat(ops.expand_dims(cond, axis=1), self.seq_len, axis=1)
        t_rep = ops.repeat(t, self.seq_len, axis=1)
        
        inputs = ops.concatenate([x_noisy, cond_rep, t_rep], axis=-1)
        
        with ops.gradient_tape.GradientTape() as tape:
            pred_noise = self.backbone(inputs)
            loss = ops.mean(keras.losses.mean_squared_error(noise, pred_noise))
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": loss}

    def generate(self, cond, num_samples):
        # Simple one-step denoising for speed in tuning
        z = keras.random.normal(shape=(num_samples, self.seq_len, self.n_features))
        cond_rep = ops.repeat(ops.expand_dims(cond, axis=1), self.seq_len, axis=1)
        t = ops.ones((num_samples, self.seq_len, 1))
        
        inputs = ops.concatenate([z, cond_rep, t], axis=-1)
        pred_noise = self.backbone(inputs)
        return z - pred_noise # Denoised signal

def train_generative_model(model_class, dataset, sliding_window_size, sliding_window_stride, 
                           latent_dim=16, epochs=200, batch_size=32, output_dir='generative_results'):
    """
    Generic training function for generative models.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model_name = model_class.__name__
    print(f"Starting training for {model_name}...")

    # Normalize Data
    mean, std = compute_mean_std(dataset)
    scaler = StdScalerTransform(mean=mean, std=std)
    
    # Extract data from dataset
    x_train = dataset.data # Tensor (N, W, C)
    x_proc = dataset.proc_data # Tensor (N, P_total)
    y_train = dataset.targets # Tensor (N, 1)
    
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
    
    # Initialize Model
    if model_name == 'ConditionalVAE':
        model = model_class(input_shape=(seq_len, feature_dim), cond_dim=cond_dim, latent_dim=latent_dim)
    elif model_name == 'TimeGAN':
        model = model_class(seq_len=seq_len, n_features=feature_dim, cond_dim=cond_dim, latent_dim=latent_dim)
    elif model_name == 'TimeGrad':
        model = model_class(seq_len=seq_len, n_features=feature_dim, cond_dim=cond_dim)
        
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)
    
    # Train
    print("Fitting model...")
    # fit expects ([x, cond], x) for VAE or similar
    model.fit([x_train_norm, condition_vector], x_train_norm, epochs=epochs, batch_size=batch_size, shuffle=True)
    
    # Save Model and Stats
    suffix = f"sw{sliding_window_size}_ss{sliding_window_stride}"
    weights_path = os.path.join(output_dir, f"{model_name}_{suffix}.weights.h5")
    model.save_weights(weights_path)
    
    scaler_stats = {
        'signal_mean': mean.numpy() if hasattr(mean, 'numpy') else mean,
        'signal_std': std.numpy() if hasattr(std, 'numpy') else std,
        'proc_min': proc_min.numpy() if hasattr(proc_min, 'numpy') else proc_min,
        'proc_max': proc_max.numpy() if hasattr(proc_max, 'numpy') else proc_max,
        'target_min': y_min.numpy() if hasattr(y_min, 'numpy') else y_min,
        'target_max': y_max.numpy() if hasattr(y_max, 'numpy') else y_max
    }
    stats_path = os.path.join(output_dir, f"{model_name}_scaler_{suffix}.joblib")
    dump(scaler_stats, stats_path)
    
    print(f"Training complete. Weights saved to {weights_path}, stats to {stats_path}")
    return model
