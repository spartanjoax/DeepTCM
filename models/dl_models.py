import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Conv2D, Conv1D, AveragePooling2D, AveragePooling1D, Flatten, Bidirectional, GRU, concatenate, Dropout, BatchNormalization, Activation, Add, Multiply, GlobalAveragePooling1D, GlobalAveragePooling2D, Lambda, Resizing, LayerNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import Huber
from keras.applications import Xception
from keras import backend as K

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
    
    model_resnet.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
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
    x = Dense(128, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    if regress:
        x = Dense(1, activation='linear', name=name+'_output')(x)
    model_resnet = Model(inputs=signal_input_resnet, outputs=x, name=name)    
    
    if regress:
        model_resnet.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
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
    x = Dense(128, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_lstm = Model(inputs=signal_input_lstm, outputs=x, name=name)    
    model_lstm.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])

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
    x = Dense(128, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    
    model_bigru = Model(inputs=signal_input_bigru, outputs=x, name=name)    
    model_bigru.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])

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
    x = Dense(128, activation='relu', name=name+'_dense2')(x)
    x = Dropout(dropout, name=name+'_drop_final_2')(x)
    x = Dense(1, activation='linear', name=name+'_output')(x)
    model_cnn = Model(inputs=signal_input_cnn, outputs=x, name=name)
    
    model_cnn.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
    
    return model_cnn

def create_ensemble(learning_r=1e-3,
                    decay=1e-5,
                    dropout=0.2,
                    xproc_shape=[1,1],
                    name='ens',
                    hidden_units=[128, 128],
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
            y = Dense(xproc_shape[1], activation='linear', name='process_output')(process_input)
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

        model_ens.compile(loss=Huber(delta=0.1), optimizer=Adam(learning_rate=learning_r, clipnorm=1.0), metrics=[root_mse])
        return model_ens
    else:
        return None
