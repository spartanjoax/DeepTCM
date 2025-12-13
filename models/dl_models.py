import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, Flatten, Bidirectional, GRU, concatenate, Dropout, BatchNormalization, Activation, Add, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D, Lambda, Resizing, LayerNormalization, Reshape, Conv1DTranspose, Concatenate, UpSampling1D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import Huber
from keras.applications import Xception
from keras import backend as K
#from keras import ops

def root_mse(y_test, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_test)))

def conv_bn_relu(x, filters, kernel_size, padding='valid', name='', line=0, decay=1e-5, signal_qty=1, is_2d=False, grouping=False):
    if not is_2d:
        x = Conv1D(filters=filters*(signal_qty if grouping else 1), kernel_size=kernel_size, groups=signal_qty if grouping else 1, padding=padding, kernel_regularizer=l2(decay), name=name+'_Conv_'+str(line))(x)
    else:
        x = Conv2D(filters=filters*(signal_qty if grouping else 1), kernel_size=kernel_size, groups=signal_qty if grouping else 1, padding=padding, kernel_regularizer=l2(decay), name=name+'_Conv_'+str(line))(x)
        
    x = BatchNormalization(name=name+'_BN_'+str(line))(x)
    x = Activation('relu', name=name+'_ReLU_'+str(line))(x)
    return x

def resnet_block(x, filters, kernel_size, name='', line=0, attention=False, downsample=False, signal_qty=1, is_2d=False):
    shortcut = x
    x = conv_bn_relu(x, filters, kernel_size, padding='same', name=name, line=line, signal_qty=signal_qty, is_2d=is_2d)
    x = conv_bn_relu(x, filters, kernel_size, padding='same', name=name, line=str(line)+'_1', signal_qty=signal_qty, is_2d=is_2d)
    if downsample:
        if not is_2d:
            shortcut = Conv1D(kernel_size=1, strides=1, filters=filters, padding="same", name=name+'_res_'+str(line)+'_down')(shortcut)
        else:
            shortcut = Conv2D(kernel_size=1, strides=2, filters=filters, padding="same", name=name+'_res_'+str(line)+'_down')(shortcut)
    if attention:
        x = Multiply(name=name+f'_mult_{line}')([x, shortcut])
    x = Add(name=name+f'_add_{line}')([x, shortcut])
    return x

def feature_denoising(x, filters, kernel_size, dropout, signal_qty=1, name='', padding='valid', is_2d=False):
    if not is_2d:
        x = Conv1D(filters=filters*signal_qty, kernel_size=5, groups=signal_qty, name=name+'_Denoise')(x)
    else:
        x = Conv2D(filters=filters*signal_qty, kernel_size=(5, 1), groups=signal_qty, name=name+'_Denoise')(x)
    x = BatchNormalization(name=name+'_Denoise_BN')(x)
    x = Activation('relu', name=name+'_Denoise_ReLU')(x) 
    x = Dropout(dropout, name=name+'_Denoise_drop')(x)
    return x

def create_xception(learning_r=1e-3,
               decay=1e-5,
               dropout=0.2,
               x_shape=[1,1],
               name='xception',
               scalogram_cwt=None):
    is_2d = scalogram_cwt is not None
    
    signal_input_resnet = Input(x_shape[1:], name=name+'_signal_input')

    height = 224
    width = x_shape[-2]
    
    resnet = Xception(include_top=False, pooling='avg', input_shape=(height, width, 3))
    
    if not is_2d:
        x = Lambda(lambda x: tf.expand_dims(x, axis=1))(signal_input_resnet)  # Add height dimension

    x = Conv2D(filters=3, kernel_size=1, padding='same', strides=1, name=name+'_ConvResize')(x if not is_2d else signal_input_resnet)
    print("Change to 3 channels:",x.shape)
    
    x = Resizing(224, width)(x)
    print("Resize:",x.shape)
    
    x = resnet(x)
    
    x = Dense(512, activation='relu', name=name+'_dense1')(x)
    x = Dropout(dropout, name=name+'_drop_final')(x)
    x = Dense(512, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_resnet = Model(inputs=signal_input_resnet, outputs=x, name=name)    
    
    model_resnet.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
    return model_resnet

def create_resnet(learning_r=1e-3,
               decay=1e-5,
               dropout=0.2,
               x_shape=[1,1],
               name='resnet',
               filters=[32, 64, 64, 64, 128, 128, 128, 256],
               kernel_size=[3, 3, 3, 3, 3, 3, 3, 3],
               pooling=[2, 2, 2, 2, 2, 2, 2, 2],
               apply_dropout=[True, False, False, False, False, False, False, False],
               apply_pooling=[True, False, False, True, False, False, True, False],
               is_resnet=[False, False, True, True, False, True, True, False],
               regress=True, 
               attention=False,
               scalogram_cwt=None):
    is_2d = scalogram_cwt is not None
    signal_qty = x_shape[-1]
    padding = 'valid' if x_shape[-1] == 1 else 'same'
    
    signal_input_resnet = Input(x_shape[1:], name=name+'_signal_input')
    
    if not is_2d:
        x = BatchNormalization(name=name+'_BN_input')(signal_input_resnet)
    
    x = feature_denoising(x if not is_2d else signal_input_resnet, filters[0], kernel_size[0], dropout if apply_dropout[0] else 0, name=name, signal_qty=signal_qty, padding=padding, is_2d=is_2d)
    
    for i in range(1, len(filters)):
        if not is_resnet[i]:
            x = conv_bn_relu(x, filters[i], kernel_size[i], name=name, line=i+1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d)
        else:
            downsample = filters[i] < filters[i-1]
            x = resnet_block(x, filters[i], kernel_size[i], name=name, line=i+1, attention=attention, downsample=downsample, signal_qty=signal_qty, is_2d=is_2d)
            
        if apply_dropout[i]:
            x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
            
        if apply_pooling[i]:
            if not is_2d:
                x = AveragePooling1D(pool_size=pooling[i], name=name+f'_pool_{i+1}')(x)
            else:
                x = AveragePooling2D(pool_size=pooling[i], name=name+f'_pool_{i+1}', padding=padding)(x)
        
    if is_2d:
        x = GlobalAveragePooling2D(name=name+'_gap')(x)
    else:
        x = GlobalAveragePooling1D(name=name+'_gap')(x)
        
    x = Dense(128, activation='relu', name=name+'_dense1')(x)
    x = Dropout(dropout, name=name+'_drop_final')(x)
    x = Dense(64, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    if regress:
        x = Dense(1, activation='linear', name=name+'_output')(x)
    model_resnet = Model(inputs=signal_input_resnet, outputs=x, name=name)    
    
    if regress:
        model_resnet.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
    return model_resnet

def create_lstm(learning_r=1e-3,
                decay=1e-5,
                dropout=0.2,
                x_shape=[1,1],
                name='LSTM',
                hidden_units=[128, 128, 128, 128, 128, 128]):
    signal_input_lstm = Input((x_shape[1], x_shape[2]), name=name+'_signal_input')  
    x = BatchNormalization(name=name+'_BN_input')(signal_input_lstm)
    
    for i in range(len(hidden_units)):
        if i == len(hidden_units)-1: #Last, no return_sequences
            x = Bidirectional(LSTM(hidden_units[i], kernel_regularizer=l2(decay)), name=name+f'_{i+1}')(x)
        else: #Rest
            x = Bidirectional(LSTM(hidden_units[i], return_sequences=True, kernel_regularizer=l2(decay)), name=name+f'_{i+1}')(x)
        x = LayerNormalization(name=name+f'_BN_{i+1}')(x)

    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)  # Add height dimension
    x = GlobalAveragePooling1D(name=name+'_gap')(x)
        
    x = Dense(128, activation='relu', name=name+'_dense1')(x)
    x = Dropout(dropout, name=name+'_drop_final')(x)
    x = Dense(64, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_lstm = Model(inputs=signal_input_lstm, outputs=x, name=name)    
    model_lstm.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])

    return model_lstm

def create_bigru(learning_r=1e-3,
                decay=1e-5,
                dropout=0.2,
                x_shape=[1,1],
                name='BiGRU',
                hidden_units=[128, 128, 128, 128, 128, 128]):
    signal_input_bigru = Input((x_shape[1], x_shape[2]), name=name+'_signal_input')
    x = BatchNormalization(name=name+'_BN_input')(signal_input_bigru)
    
    for i in range(len(hidden_units)):
        if i == len(hidden_units)-1: #Last, no return_sequences
            x = Bidirectional(GRU(hidden_units[i], kernel_regularizer=l2(decay)), name=name+f'_{i+1}')(x)
        else: #Rest
            x = Bidirectional(GRU(hidden_units[i], return_sequences=True, kernel_regularizer=l2(decay)), name=name+f'_{i+1}')(x)
        x = LayerNormalization(name=name+f'_LN_{i+1}')(x)
        
    x = Lambda(lambda x: tf.expand_dims(x, axis=1))(x)  # Add height dimension
    x = GlobalAveragePooling1D(name=name+'_gap')(x)
        
    x = Dense(128, activation='relu', name=name+'_dense1')(x)
    x = Dropout(dropout, name=name+'_drop_final')(x)
    x = Dense(64, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_bigru = Model(inputs=signal_input_bigru, outputs=x, name=name)    
    model_bigru.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])

    return model_bigru

def create_cnn(learning_r=1e-3,
               decay=1e-5,
               dropout=0.2,
               x_shape=[1,1],
               name='CNN',
               filters=[32, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256],
               kernel_size=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
               pooling=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
               apply_dropout=[True, False, False, False, False, False, False, False, False, False, False],
               apply_pooling=[True, False, True, False, True, False, False, True, False, True, False],
               scalogram_cwt=None):  
    is_2d = scalogram_cwt is not None
    padding = 'valid' if x_shape[2] == 1 else 'same'
    signal_qty = x_shape[-1]

    signal_input_cnn = Input(x_shape[1:], name=name+'_signal_input')

    if not is_2d:
        x = BatchNormalization(name=name+'_BN_input')(signal_input_cnn)
    
    x = conv_bn_relu(x if not is_2d else signal_input_cnn, filters[0], kernel_size[0], name=name, line=1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d, grouping=True)
    if apply_pooling[0]:
        if not is_2d:
            x = AveragePooling1D(pooling[0], name=name+'_pool_1')(x)
        else:
            x = AveragePooling2D(pool_size=pooling[0], name=name+'_pool_1', padding=padding)(x)
    if apply_dropout[0]:
        x = Dropout(dropout, name=name+'_drop_1')(x)
    
    for i in range(1, len(filters)):
        x = conv_bn_relu(x, filters[i], kernel_size[i], name=name, line=i+1, decay=decay, padding=padding, signal_qty=signal_qty, is_2d=is_2d)
        if apply_pooling[i]:
            if not is_2d:
                x = AveragePooling1D(pool_size=pooling[i], name=name+f'_pool_{i+1}')(x)
            else:
                x = AveragePooling2D(pool_size=pooling[i], name=name+f'_pool_{i+1}', padding=padding)(x)
        if apply_dropout[i]:
            x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
        
    if is_2d:
        x = GlobalAveragePooling2D(name=name+'_gap')(x)
    else:
        x = GlobalAveragePooling1D(name=name+'_gap')(x)
        
    x = Dense(128, activation='relu', name=name+'_dense1')(x)
    x = Dropout(dropout, name=name+'_drop_final')(x)
    x = Dense(64, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    model_cnn = Model(inputs=signal_input_cnn, outputs=x, name=name)
    
    model_cnn.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
    return model_cnn

def create_ensemble(learning_r=1e-3,
                    decay=1e-5,
                    dropout=0.2,
                    xproc_shape=[1,1],
                    name='ens',
                    hidden_units=[128, 64],
                    model_list=[],
                    proc_input=False,
                    freeze_base=True):
    if len(model_list) > 0:
        if freeze_base:
            for i in range(len(model_list)):
                model = model_list[i]
                for layer in model.layers:
                    layer.trainable = False
    
        if proc_input:
            process_input = Input(shape=(xproc_shape[1],), name='process_input')
            y = Dense(8, activation='relu', name=f'proc_dense')(process_input)
            y = BatchNormalization(name=f'proc_BN')(y)
            y = Dense(4, activation='linear', name='proc_output')(y)
            model_process_input = Model(inputs=process_input, outputs=y, name='process_data')

        ensemble_inputs = [model.input for model in model_list]
        if proc_input:
            ensemble_inputs.append(model_process_input.input)
        ensemble_outputs = [model.layers[-1].output for model in model_list]
        if proc_input:
            ensemble_outputs.append(model_process_input.output)

        merge = concatenate(ensemble_outputs, name=name+'_concat')
        x = BatchNormalization(name=name+'_concat_BN')(merge)
        
        for i in range(len(hidden_units)):
            x = Dense(hidden_units[i], activation='relu', kernel_regularizer=l2(decay), name=name+f'_{i+1}')(x)
            x = BatchNormalization(name=name+f'_BN_{i+1}')(x)
            
        if dropout != 0:
            x = Dropout(dropout, name=name+f'_drop_{i+1}')(x)
            
        x = Dense(1, activation='linear', name=name+'_output')(x)
        model_ens = Model(inputs=ensemble_inputs, outputs=x)

        model_ens.compile(loss=Huber(delta=0.5), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
        return model_ens
    else:
        return None

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class CVAE(Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs):
        if isinstance(inputs, list):
            x, y = inputs
        else:
             x = inputs
             y = None # Should not happen if correctly used

        z_mean, z_log_var, z = self.encoder([x, y])
        reconstruction = self.decoder([z, y])
        return reconstruction

    def train_step(self, data):
        # Data is a tuple (x, y) because we passed (x, y) to fit
        # We need to concatenate x and y for the conditional part
        if isinstance(data, tuple):
             x, y = data  # x is [batch, len, chan], y is [batch, 1]
        else:
             x = data
             y = None # Should not happen in conditional VAE

        with tf.GradientTape() as tape:
            # Encoder output
            z_mean, z_log_var, z = self.encoder([x, y])
            
            # Decoder output
            reconstruction = self.decoder([z, y])
            
            # Reconstruction loss (MSE)
            # Sum over all dimensions
            hwc = tf.cast(tf.reduce_prod(tf.shape(x)[1:]), tf.float32)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(x, reconstruction), axis=1
                )
            )
            
            # KL divergence
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            total_loss = reconstruction_loss + self.beta * kl_loss
            
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

def create_cvae(input_shape, label_dim=1, latent_dim=16, filters=[32, 64, 128], kernel_size=[3, 3, 3], strides=[2, 2, 2], name='cvae'):
    # Encoder
    encoder_inputs = Input(shape=input_shape, name='encoder_input')
    label_input = Input(shape=(label_dim,), name='label_input')
    
    # Conditional input: Tile label and concatenate with input signal 
    # Label is (Batch, 1), Signal is (Batch, Len, Chan)
    # We repeat label along length
    
    # 1. Expand label to (Batch, 1, 1)
    # label_reshaped = Reshape((1, 1))(label_input) 
    # 2. Tile to (Batch, Len, 1) - Doing this dynamically might be tricky with symbolic tensor size
    # Alternatively, use a Dense layer to project label to match spatial dims or just concat at any level.
    # A standard way for 1D signals is to expand and tile.
    
    # Let's simple concatenate the label as a channel after repeating?
    # Or, following standard practice for CVAE on images: process image and label separately then merge, or tile label channel-wise.
    # Tiling is better for conv nets.
    
    seq_len = input_shape[0]
    
    # Tile label: (Batch, 1) -> (Batch, SeqLen, 1)
    label_tiled = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, seq_len, 1]))(label_input)
    
    x = Concatenate(axis=-1)([encoder_inputs, label_tiled]) # (Batch, SeqLen, Chan+1)
    
    for i in range(len(filters)):
        x = Conv1D(filters[i], kernel_size[i], strides=strides[i], padding="same", activation="relu", name=f"enc_conv_{i}")(x)
        # x = BatchNormalization()(x)
        
    shape_before_flatten = x.shape[1:]
    
    x = Flatten()(x)
    x = Dense(latent_dim * 2, activation="relu")(x) # Optional dense
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    
    encoder = Model([encoder_inputs, label_input], [z_mean, z_log_var, z], name="encoder")
    
    # Decoder
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")
    # Label input is reused or new input
    label_input_dec = Input(shape=(label_dim,), name='label_input_dec')
    
    # Concatenate z and label
    x = Concatenate(axis=-1)([latent_inputs, label_input_dec])
    
    # Project back to flattening shape
    # Need to calculate flat dimensions from shape_before_flatten
    flat_dim = shape_before_flatten[0] * shape_before_flatten[1]
    
    x = Dense(flat_dim, activation="relu")(x)
    x = Reshape(shape_before_flatten)(x)
    
    # Reverse conv layers
    for i in range(len(filters) - 1, -1, -1):
        if i == 0:
            # Last layer restores original depth but keep activation linear or sigmoid?
            # VAE usually reconstructs unscaled or scaled 0-1.
            # We assume inputs are standardized (mean 0 std 1), so linear output is okay.
            x = Conv1DTranspose(input_shape[-1], kernel_size[i], strides=strides[i], padding="same", activation="linear", name="dec_output")(x)
        else:
            x = Conv1DTranspose(filters[i-1], kernel_size[i], strides=strides[i], padding="same", activation="relu", name=f"dec_conv_{i}")(x)
            # x = BatchNormalization()(x)
            
    # Crop to original sequence length if mismatch
    # Output shape is (Batch, NewSeqLen, Channels). Slicing works if NewSeqLen >= SeqLen
    if seq_len is not None:
         x = Lambda(lambda t: t[:, :seq_len, :], name='decoder_cropping')(x)
            
    decoder_outputs = x
    decoder = Model([latent_inputs, label_input_dec], decoder_outputs, name="decoder")
    
    cvae = CVAE(encoder, decoder, name=name)
    
    return cvae
