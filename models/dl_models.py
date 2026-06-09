################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""Keras model zoo for TCM.

Defines all expert-designed architectures (CNN, ResNet, BiGRU, LSTM,
CNN_LSTM, Transformer) and their FiLM-conditioned multimodal variants.
Custom attention layers (CBAM channel/spatial attention) and the Run-level
aggregation wrapper are also included.
Main entry points: get_model() and create_proc_model().
"""
import os
import keras.ops as ops
import keras
from keras.models import Model
from keras.layers import (
    Input,
    LSTM,
    Dense,
    Conv2D,
    Conv1D,
    AveragePooling2D,
    AveragePooling1D,
    MaxPooling2D,
    MaxPooling1D,
    Flatten,
    Bidirectional,
    GRU,
    concatenate,
    Dropout,
    BatchNormalization,
    Activation,
    Add,
    Multiply,
    GlobalAveragePooling1D,
    GlobalAveragePooling2D,
    Lambda,
    Resizing,
    LayerNormalization,
    Reshape,
    Concatenate,
    TimeDistributed,
    MultiHeadAttention,
    Embedding,
    Layer
)
from keras import backend as K

from keras.optimizers import Adam
from keras.regularizers import l2
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.applications import Xception
import logging
import gc
import torch
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)

root_mse = RootMeanSquaredError(name="root_mse")

class MaskedPool1D(keras.layers.Layer):
    """Mask-aware pooling over the window (time) axis for run-level models.

    Accepts ``[x_embedded, raw_signal]`` where:
    - ``x_embedded``: ``(batch, n_r_max, F)`` — post-backbone embeddings.
    - ``raw_signal`` : ``(batch, n_r_max, *window_dims)`` — original padded signal.

    A window is treated as padding when ALL values are exactly 0.  This is
    reliable because real windows always have non-zero values after preprocessing.

    Unlike Keras's built-in ``Masking + GlobalAveragePooling1D``, this works
    correctly for any input rank (3-D, 4-D, 5-D ...) without mask-shape
    mismatches.
    """

    def __init__(self, mode: str = "mean", **kwargs):
        """Initialise MaskedPool1D.

        Args:
            mode: Pooling mode — ``"mean"`` (default) or ``"max"``.
        """
        super().__init__(**kwargs)
        if mode not in ("mean", "max"):
            raise ValueError(f"MaskedPool1D mode must be 'mean' or 'max', got '{mode}'")
        self.mode = mode

    def call(self, inputs):
        """Apply masked pooling, ignoring zero-padded timesteps.

        Args:
            inputs: Tuple of (x, raw_signal) where x has shape (B, T, F)
                and raw_signal has shape (B, T, ...).

        Returns:
            Pooled tensor of shape (B, F).
        """
        x, raw_signal = inputs
        sig_shape = keras.ops.shape(raw_signal)
        flat = keras.ops.reshape(raw_signal, (sig_shape[0], sig_shape[1], -1))
        is_real = keras.ops.any(flat != 0.0, axis=-1)
        mask_f = keras.ops.cast(is_real, x.dtype)
        mask_exp = keras.ops.expand_dims(mask_f, axis=-1)

        if self.mode == "max":
            neg_inf = keras.ops.full_like(x, -1e9)
            x_masked = keras.ops.where(
                keras.ops.cast(mask_exp, "bool"), x, neg_inf
            )
            return keras.ops.max(x_masked, axis=1)
        else:
            x_masked = x * mask_exp
            x_sum = keras.ops.sum(x_masked, axis=1)
            valid_n = keras.ops.maximum(
                keras.ops.sum(mask_f, axis=1, keepdims=True),
            )
            return x_sum / valid_n

    def compute_output_shape(self, input_shape):
        """Return output shape (B, F) after pooling over the time axis."""
        x_shape = input_shape[0]
        return (x_shape[0], x_shape[-1])

    def get_config(self):
        """Return serialisable layer configuration dict."""
        cfg = super().get_config()
        cfg["mode"] = self.mode
        return cfg


class CleanMemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        Aggressive Garbage Collection to prevent Memory Leaks in Keras 3 + PyTorch.

        Args:
            epoch (int): The index of the epoch.
            logs (dict): Dictionary of logs.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_callbacks(
    patience=10,
    min_delta=1e-4,
    monitor="val_loss",
    mode="min",
    reduce_lr=False,
    cleanup_memory=True,
    patience_lr=5,
    factor_lr=0.5,
    min_lr=1e-6,
    disable_es=False,
    verbose=1,
):
    """
    Creates a list of standard callbacks for training.

    Args:
        patience (int): Patience for EarlyStopping.
        min_delta (float): Minimum change to qualify as an improvement.
        monitor (str): Metric to monitor.
        mode (str): Optimization mode ('min' or 'max').
        reduce_lr (bool): If True, adds ReduceLROnPlateau.
        cleanup_memory (bool): If True, adds CleanMemoryCallback.
        patience_lr (int): Patience for ReduceLROnPlateau.
        factor_lr (float): Factor for ReduceLROnPlateau.
        min_lr (float): Minimum learning rate.
        disable_es (bool): If True, EarlyStopping is NOT added. Use for final
            retraining with a fixed epoch budget where no validation set exists
            and ES on train loss would produce undefined stopping behaviour.
        verbose (int): Verbosity level passed to EarlyStopping and
            ReduceLROnPlateau. Defaults to 1.

    Returns:
        list: List of Keras callbacks.
    """
    callbacks = []

    if not disable_es:
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                start_from_epoch=0,
                min_delta=min_delta,
                mode=mode,
                verbose=verbose,
            )
        )

    if reduce_lr:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=monitor,
                factor=factor_lr,
                patience=patience_lr,
                min_lr=min_lr,
                mode=mode,
                verbose=verbose,
            )
        )

    if cleanup_memory:
        callbacks.append(CleanMemoryCallback())

    return callbacks


def cbam_block(x, ratio=8, is_2d=False, name="cbam"):
    """Apply a Convolutional Block Attention Module (CBAM) to a Keras tensor.

    Combines channel attention (global avg/max pooling + MLP) and spatial
    attention (avg/max across channels + 1×7 conv) to reweight the input.

    Args:
        x: Input Keras tensor of shape (B, T, C) or (B, H, W, C).
        ratio (int): Channel-reduction ratio for the MLP bottleneck.
        is_2d (bool): Use 2-D pooling and convolution when True.
        name (str): Prefix for all layer names.

    Returns:
        Keras tensor with the same shape as ``x``, scaled by the attention maps.
    """
    channel_axis = -1 if K.image_data_format() == "channels_last" else 1

    filters = x.shape[channel_axis]

    if is_2d:
        avg_pool = GlobalAveragePooling2D(keepdims=True, name=f"{name}_ca_avg_pool")(x)
        max_pool = keras.layers.GlobalMaxPooling2D(
            keepdims=True, name=f"{name}_ca_max_pool"
        )(x)
    else:
        avg_pool = GlobalAveragePooling1D(keepdims=True, name=f"{name}_ca_avg_pool")(x)
        max_pool = keras.layers.GlobalMaxPooling1D(
            keepdims=True, name=f"{name}_ca_max_pool"
        )(x)

    mlp_1 = Dense(
        max(filters // ratio, 4),
        activation="relu",
        kernel_initializer="he_normal",
        use_bias=True,
        name=f"{name}_ca_mlp_1",
    )
    mlp_2 = Dense(
        filters, kernel_initializer="he_normal", use_bias=True, name=f"{name}_ca_mlp_2"
    )

    avg_out = mlp_2(mlp_1(avg_pool))
    max_out = mlp_2(mlp_1(max_pool))

    channel_out = Add(name=f"{name}_ca_add")([avg_out, max_out])
    channel_out = Activation("sigmoid", name=f"{name}_ca_sigmoid")(channel_out)

    x_ca = Multiply(name=f"{name}_ca_mult")([x, channel_out])

    def channel_pool_output_shape(input_shape):
        """Compute the output shape of a channel-wise global-pooling layer."""
        return tuple(list(input_shape)[:-1] + [1])

    avg_pool_s = Lambda(
        lambda y: ops.mean(y, axis=channel_axis, keepdims=True),
        output_shape=channel_pool_output_shape,
        name=f"{name}_sa_avg_pool",
    )(x_ca)
    max_pool_s = Lambda(
        lambda y: ops.max(y, axis=channel_axis, keepdims=True),
        output_shape=channel_pool_output_shape,
        name=f"{name}_sa_max_pool",
    )(x_ca)

    concat = Concatenate(axis=channel_axis, name=f"{name}_sa_concat")(
        [avg_pool_s, max_pool_s]
    )

    if is_2d:
        sa_out = Conv2D(
            1,
            (7, 7),
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=f"{name}_sa_conv",
        )(concat)
    else:
        sa_out = Conv1D(
            1,
            7,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            name=f"{name}_sa_conv",
        )(concat)

    x_sa = Multiply(name=f"{name}_sa_mult")([x_ca, sa_out])

    return x_sa


def get_optimizer(learning_rate, opt_name="adam"):
    """Build a Keras optimizer with gradient clipping.

    Args:
        learning_rate (float): Optimizer learning rate.
        opt_name (str): One of ``'adam'``, ``'sgd'``, or
            ``'adam_weight_decay'``. Defaults to ``'adam'``.

    Returns:
        keras.Optimizer: Configured optimizer with ``clipnorm=1.0``.
    """
    if opt_name == "sgd":
        return keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=1.0)
    elif opt_name == "adam_weight_decay":
        return keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    else:
        return keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


def conv_bn_relu(
    x,
    filters,
    kernel_size,
    padding="valid",
    name="",
    line=0,
    decay=1e-5,
    signal_qty=1,
    is_2d=False,
    grouping=False,
    activation=True,
):
    """Apply Conv1D/Conv2D → BatchNorm → optional ReLU block.

    Args:
        x: Input Keras tensor.
        filters (int): Number of convolution filters.
        kernel_size (int or tuple): Convolution kernel size.
        padding (str): ``'valid'`` or ``'same'``. Defaults to ``'valid'``.
        name (str): Prefix for layer names.
        line (int): Block index suffix for layer names.
        decay (float): L2 weight-decay coefficient.
        signal_qty (int): Group size for grouped convolution.
        is_2d (bool): Use Conv2D/BN2D when True.
        grouping (bool): Enable grouped-convolution mode.
        activation (bool): Apply ReLU after BatchNorm when True.

    Returns:
        Keras tensor.
    """
    if not is_2d:
        x = Conv1D(
            filters=filters * (signal_qty if grouping else 1),
            kernel_size=kernel_size,
            groups=signal_qty if grouping else 1,
            padding=padding,
            kernel_regularizer=l2(decay),
            name=name + "_Conv_" + str(line),
        )(x)
    else:
        x = Conv2D(
            filters=filters * (signal_qty if grouping else 1),
            kernel_size=kernel_size,
            groups=signal_qty if grouping else 1,
            padding=padding,
            kernel_regularizer=l2(decay),
            name=name + "_Conv_" + str(line),
        )(x)

    x = BatchNormalization(name=name + "_BN_" + str(line))(x)
    if activation:
        x = Activation("relu", name=name + "_ReLU_" + str(line))(x)
    return x


def resnet_block(
    x,
    filters,
    kernel_size,
    name="",
    line=0,
    attention=False,
    downsample=False,
    signal_qty=1,
    is_2d=False,
):
    """Build a single ResNet residual block with optional CBAM attention and downsampling.

    Args:
        x: Input Keras tensor.
        filters (int): Number of filters in each convolution.
        kernel_size (int): Kernel size for the main convolutions.
        name (str): Prefix for layer names.
        line (int): Block index suffix for layer names.
        attention (bool): Insert a CBAM attention gate after the second conv.
        downsample (bool): Project the shortcut when channel dimensions differ.
        signal_qty (int): Group size for grouped convolution.
        is_2d (bool): Use 2-D operations when True.

    Returns:
        Keras tensor after residual add and ReLU activation.
    """
    shortcut = x
    x = conv_bn_relu(
        x,
        filters,
        kernel_size,
        padding="same",
        name=name,
        line=line,
        signal_qty=signal_qty,
        is_2d=is_2d,
    )
    x = conv_bn_relu(
        x,
        filters,
        kernel_size,
        padding="same",
        name=name,
        line=str(line) + "_1",
        signal_qty=signal_qty,
        is_2d=is_2d,
        activation=False,  # Skip activation inside the block
    )
    if downsample:
        if not is_2d:
            shortcut = Conv1D(
                kernel_size=1,
                strides=1,
                filters=filters,
                padding="same",
                name=name + "_res_" + str(line) + "_down",
            )(shortcut)
        else:
            shortcut = Conv2D(
                kernel_size=1,
                strides=2,
                filters=filters,
                padding="same",
                name=name + "_res_" + str(line) + "_down",
            )(shortcut)

    if shortcut.shape[-1] != filters:
        if not is_2d:
            shortcut = Conv1D(
                kernel_size=1,
                strides=1,
                filters=filters,
                padding="same",
                name=name + "_res_" + str(line) + "_proj",
            )(shortcut)
        else:
            shortcut = Conv2D(
                kernel_size=1,
                strides=1,
                filters=filters,
                padding="same",
                name=name + "_res_" + str(line) + "_proj",
            )(shortcut)

    if attention:
        x = cbam_block(x, is_2d=is_2d, name=name + f"_cbam_{line}")

    x = Add(name=name + f"_add_{line}")([x, shortcut])
    x = Activation("relu", name=name + f"_relu_out_{line}")(x)
    return x


def feature_denoising(
    x,
    filters,
    kernel_size,
    dropout,
    signal_qty=1,
    name="",
    padding="valid",
    is_2d=False,
):
    """Add a denoising convolution block before the main feature extractor.

    Args:
        x: Input Keras tensor.
        filters (int): Number of convolution filters.
        kernel_size (int): Kernel size (overridden to 5 internally).
        dropout (float): Dropout rate applied before the convolution.
        signal_qty (int): Group size for grouped convolution. Defaults to 1.
        name (str): Prefix for layer names.
        padding (str): ``'valid'`` or ``'same'``. Defaults to ``'valid'``.
        is_2d (bool): Use 2-D operations when True.

    Returns:
        Keras tensor after convolution and residual shortcut.
    """
    if not is_2d:
        x = Conv1D(
            filters=filters * signal_qty,
            kernel_size=5,
            groups=signal_qty,
            name=name + "_Denoise",
        )(x)
    else:
        x = Conv2D(
            filters=filters * signal_qty,
            kernel_size=(5, 1),
            groups=signal_qty,
            name=name + "_Denoise",
        )(x)
    x = BatchNormalization(name=name + "_Denoise_BN")(x)
    x = Activation("relu", name=name + "_Denoise_ReLU")(x)
    x = Dropout(dropout, name=name + "_Denoise_drop")(x)
    return x


def create_resnet(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    x_shape=[1, 1],
    name="resnet",
    filters=[32, 64, 64, 128, 128],
    kernel_size=[7, 5, 5, 3, 3],
    apply_dropout=[True, False, False, False, False],
    apply_pooling=[True, False, False, True, False],
    is_resnet=[False, True, True, True, True],
    regress=True,
    attention=False,
    scalogram_cwt=None,
    pooling_type="avg",
    **kwargs,
):
    """
    Creates a custom ResNet-like model.

    Args:
        learning_r (float): Learning rate.
        decay (float): Weight decay.
        dropout (float): Dropout rate.
        x_shape (tuple): Input shape.
        name (str): Model name.
        filters (list): List of filter counts for each convolutional block.
        kernel_size (list): List of kernel sizes for each convolutional block.
        apply_dropout (list): List of booleans indicating whether to apply dropout after each block.
        apply_pooling (list): List of booleans indicating whether to apply pooling after each block.
        is_resnet (list): List of booleans indicating whether a block is a ResNet block.
        regress (bool): If True, the model outputs a single regression value.
        attention (bool): If True, adds attention mechanism to ResNet blocks.
        scalogram_cwt (str): If not None, assumes 2D inputs from scalograms.
        pooling_type (str): 'max' or 'avg'. Defaults to 'max'.

    Returns:
        keras.Model: Compiled Keras model.
    """
    is_2d = scalogram_cwt is not None
    signal_qty = x_shape[-1]
    padding = "valid" if x_shape[-1] == 1 else "same"

    def start_pooling(x, pool_size, name_p, is_2d_p):
        """Select and apply a 1-D or 2-D pooling layer based on pooling_type."""
        if pooling_type == "max":
            if not is_2d_p:
                return MaxPooling1D(pool_size=pool_size, name=name_p)(x)
            else:
                return MaxPooling2D(pool_size=pool_size, name=name_p, padding=padding)(
                    x
                )
        else:
            if not is_2d_p:
                return AveragePooling1D(pool_size=pool_size, name=name_p)(x)
            else:
                return AveragePooling2D(
                    pool_size=pool_size, name=name_p, padding=padding
                )(x)

    signal_input_resnet = Input(x_shape[1:], name=name + "_signal_input")

    if not is_2d:
        x = BatchNormalization(name=name + "_BN_input")(signal_input_resnet)

    x = feature_denoising(
        x if not is_2d else signal_input_resnet,
        filters[0],
        kernel_size[0],
        dropout if apply_dropout[0] else 0,
        name=name,
        signal_qty=signal_qty,
        padding=padding,
        is_2d=is_2d,
    )

    for i in range(1, len(filters)):
        if not is_resnet[i]:
            x = conv_bn_relu(
                x,
                filters[i],
                kernel_size[i],
                name=name,
                line=i + 1,
                decay=decay,
                padding=padding,
                signal_qty=signal_qty,
                is_2d=is_2d,
            )
        else:
            downsample = filters[i] < filters[i - 1]
            x = resnet_block(
                x,
                filters[i],
                kernel_size[i],
                name=name,
                line=i + 1,
                attention=attention,
                downsample=downsample,
                signal_qty=signal_qty,
                is_2d=is_2d,
            )

        if apply_dropout[i]:
            x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)

        if apply_pooling[i]:
            x = start_pooling(x, 2, name + f"_pool_{i+1}", is_2d)

    if is_2d:
        x = GlobalAveragePooling2D(name=name + "_gap")(x)
    else:
        x = GlobalAveragePooling1D(name=name + "_gap")(x)

    x = Dense(128, activation="relu", name=name + "_dense1")(x)
    x = Dropout(dropout, name=name + "_drop_final")(x)
    x = Dense(64, activation="relu", name=name + "_dense2")(x)
    x = Dropout(dropout, name=name + "_drop_final_2")(x)
    if regress:
        x = Dense(1, activation="linear", name=name + "_output")(x)
    model_resnet = Model(inputs=signal_input_resnet, outputs=x, name=name)

    if regress:
        model_resnet.compile(
            loss=MeanSquaredError(),
            optimizer=get_optimizer(learning_r),
            metrics=[root_mse],
        )

    return model_resnet


@keras.saving.register_keras_serializable(package="Custom", name="FastTorchLSTM")
class FastTorchLSTM(Layer):
    def __init__(
        self,
        units,
        return_sequences=False,
        bidirectional=False,
        dropout=0.0,
        decay=0.0,
        **kwargs,
    ):
        """Initialise FastTorchLSTM.

        Args:
            units (int): Number of LSTM hidden units.
            return_sequences (bool): Return the full output sequence when True.
            bidirectional (bool): Use bidirectional LSTM when True.
            dropout (float): Dropout rate (stored for config serialisation;
                dropout is applied externally). Defaults to 0.0.
            decay (float): L2 regularisation weight added via ``add_loss``.
            **kwargs: Forwarded to ``keras.Layer.__init__``.
        """
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.decay = decay
        self.lstm = None

    def build(self, input_shape):
        """Build the dense layers used to compute channel attention weights."""
        input_dim = input_shape[-1]

        self.lstm = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=self.units,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        """Return the output shape (same as input shape)."""
        batch = input_shape[0]
        timesteps = input_shape[1]
        out_dim = self.units * (2 if self.bidirectional else 1)
        if self.return_sequences:
            return (batch, timesteps, out_dim)
        else:
            return (batch, out_dim)

    def call(self, inputs, training=False):
        """Run the LSTM forward pass.

        Args:
            inputs: Float tensor of shape (B, T, F).
            training (bool): Training mode flag.

        Returns:
            torch.Tensor: Last hidden state of shape (B, H) when
            ``return_sequences=False``, or full sequence (B, T, H) when True.
        """
        output, (hn, cn) = self.lstm(inputs)

        if self.decay > 0.0:
            reg_loss = 0.0
            for param in self.lstm.parameters():
                reg_loss += torch.sum(param**2)
            self.add_loss(self.decay * reg_loss)

        if self.return_sequences:
            return output
        else:
            # Return last hidden state
            # hn shape: (num_directions, batch, hidden_size)
            if self.bidirectional:
                # Concatenate Forward(T) and Backward(0)
                # hn[-2] is forward, hn[-1] is backward
                return torch.cat((hn[-2], hn[-1]), dim=-1)
            else:
                return hn[-1]

    def get_config(self):
        """Return serialisable layer configuration dict."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "return_sequences": self.return_sequences,
                "bidirectional": self.bidirectional,
                "dropout": self.dropout,
                "decay": self.decay,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="Custom", name="FastTorchGRU")
class FastTorchGRU(Layer):
    def __init__(
        self,
        units,
        return_sequences=False,
        bidirectional=False,
        dropout=0.0,
        decay=0.0,
        **kwargs,
    ):
        """Initialise FastTorchGRU.

        Args:
            units (int): Number of GRU hidden units.
            return_sequences (bool): Return the full output sequence when True.
            bidirectional (bool): Use bidirectional GRU when True.
            dropout (float): Dropout rate (stored for config serialisation).
            decay (float): L2 regularisation weight added via ``add_loss``.
            **kwargs: Forwarded to ``keras.Layer.__init__``.
        """
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.decay = decay
        self.gru = None

    def build(self, input_shape):
        """Build the convolutional layer used to compute the spatial attention map."""
        input_dim = input_shape[-1]

        self.gru = torch.nn.GRU(
            input_size=input_dim,
            hidden_size=self.units,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        self.built = True

    def compute_output_shape(self, input_shape):
        """Return the output shape (same as input shape)."""
        batch = input_shape[0]
        timesteps = input_shape[1]
        out_dim = self.units * (2 if self.bidirectional else 1)
        if self.return_sequences:
            return (batch, timesteps, out_dim)
        else:
            return (batch, out_dim)

    def call(self, inputs, training=False):
        """Run the GRU forward pass.

        Args:
            inputs: Float tensor of shape (B, T, F).
            training (bool): Training mode flag.

        Returns:
            torch.Tensor: Last hidden state (B, H) or full sequence (B, T, H).
        """
        output, hn = self.gru(inputs)

        if self.decay > 0.0:
            reg_loss = 0.0
            for param in self.gru.parameters():
                reg_loss += torch.sum(param**2)
            self.add_loss(self.decay * reg_loss)

        if self.return_sequences:
            return output
        else:
            if self.bidirectional:
                return torch.cat((hn[-2], hn[-1]), dim=-1)
            else:
                return hn[-1]

    def get_config(self):
        """Return serialisable layer configuration dict."""
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "return_sequences": self.return_sequences,
                "bidirectional": self.bidirectional,
                "dropout": self.dropout,
                "decay": self.decay,
            }
        )
        return config


def create_lstm(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    x_shape=[1, 1],
    name="LSTM",
    hidden_units=[64, 64],
    regress=True,
    bidirectional=True,
    head_units=None,
    input_bn=True,
    layer_norm=True,
    **kwargs,
):
    """
    Creates an LSTM model suitable for time-series data.
    Defaults adjusted to 2 layers of 64 units (Kumar et al. 2022) to prevent overfitting.

    Args:
        learning_r (float): Learning rate.
        decay (float): Weight decay.
        dropout (float): Dropout rate.
        x_shape (tuple): Input shape (batch_size, timesteps, features).
        name (str): Model name.
        hidden_units (list): List of hidden units for each LSTM layer.
        bidirectional (bool): If True (default), wrap each LSTM in Bidirectional.
        head_units (list | None): Dense head sizes before output, e.g. [128, 64].
            None (default) uses [128, 64] for backward compatibility.
        input_bn (bool): Apply BatchNorm on input signal. Default True.
        layer_norm (bool): Apply LayerNorm after each RNN layer. Default True.
        regress (bool): If True, add a regression head and compile the model.
            Set False to return a feature extractor only.

    Returns:
        keras.Model: Compiled Keras model.
    """
    if head_units is None:
        head_units = [128, 64]

    _is_2d_input = len(x_shape) == 4
    if _is_2d_input:
        signal_input_lstm = Input((x_shape[1], x_shape[2], x_shape[3]), name=name + "_signal_input")
        x = Reshape((x_shape[1], x_shape[2] * x_shape[3]), name=name + "_freq_flatten")(signal_input_lstm)
    else:
        signal_input_lstm = Input((x_shape[1], x_shape[2]), name=name + "_signal_input")
        x = signal_input_lstm
    if input_bn:
        x = BatchNormalization(name=name + "_BN_input")(x)

    is_torch = os.environ.get("KERAS_BACKEND") == "torch"

    for i in range(len(hidden_units)):
        ret_seq = (i < len(hidden_units) - 1)
        if is_torch:
            x = FastTorchLSTM(
                hidden_units[i],
                return_sequences=ret_seq,
                bidirectional=bidirectional,
                decay=decay,
                name=name + f"_{i+1}",
            )(x)
        else:
            lstm_layer = LSTM(
                hidden_units[i],
                return_sequences=ret_seq,
                kernel_regularizer=l2(decay),
                unroll=False,
            )
            if bidirectional:
                x = Bidirectional(lstm_layer, name=name + f"_{i+1}")(x)
            else:
                lstm_layer._name = name + f"_{i+1}"
                x = lstm_layer(x)

        if layer_norm:
            x = LayerNormalization(name=name + f"_LN_{i+1}")(x)

    x = Reshape((1, -1), name=name + "_reshape")(x)  # Add height dimension safely
    x = GlobalAveragePooling1D(name=name + "_gap")(x)

    if regress:
        for i, units in enumerate(head_units):
            x = Dense(units, activation="relu", name=name + f"_dense{i+1}")(x)
            x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)
        x = Dense(1, activation="linear", name=name + "_output")(x)

    model_lstm = Model(inputs=signal_input_lstm, outputs=x, name=name)
    if regress:
        model_lstm.compile(
            loss=MeanSquaredError(),
            optimizer=get_optimizer(learning_r),
            metrics=[root_mse],
        )

    return model_lstm


def create_bigru(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    x_shape=[1, 1],
    name="BiGRU",
    hidden_units=[64, 64],
    regress=True,
    bidirectional=True,
    head_units=None,
    input_bn=True,
    layer_norm=True,
    **kwargs,
):
    """
    Creates a GRU model suitable for time-series data.
    Defaults adjusted to 2 layers of 64 units to prevent overfitting.

    Args:
        learning_r (float): Learning rate.
        decay (float): Weight decay.
        dropout (float): Dropout rate.
        x_shape (tuple): Input shape (batch_size, timesteps, features).
        name (str): Model name.
        hidden_units (list): List of hidden units for each GRU layer.
        bidirectional (bool): If True (default), wrap each GRU in Bidirectional.
        head_units (list | None): Dense head sizes before output, e.g. [128, 64].
            None (default) uses [128, 64] for backward compatibility.
        input_bn (bool): Apply BatchNorm on input signal. Default True.
        layer_norm (bool): Apply LayerNorm after each RNN layer. Default True.
        regress (bool): If True, add a regression head and compile the model.
            Set False to return a feature extractor only.

    Returns:
        keras.Model: Compiled Keras model.
    """
    if head_units is None:
        head_units = [128, 64]

    _is_2d_input = len(x_shape) == 4
    if _is_2d_input:
        signal_input_bigru = Input((x_shape[1], x_shape[2], x_shape[3]), name=name + "_signal_input")
        x = Reshape((x_shape[1], x_shape[2] * x_shape[3]), name=name + "_freq_flatten")(signal_input_bigru)
    else:
        signal_input_bigru = Input((x_shape[1], x_shape[2]), name=name + "_signal_input")
        x = signal_input_bigru
    if input_bn:
        x = BatchNormalization(name=name + "_BN_input")(x)

    is_torch = os.environ.get("KERAS_BACKEND") == "torch"

    for i in range(len(hidden_units)):
        ret_seq = (i < len(hidden_units) - 1)
        if is_torch:
            x = FastTorchGRU(
                hidden_units[i],
                return_sequences=ret_seq,
                bidirectional=bidirectional,
                decay=decay,
                name=name + f"_{i+1}",
            )(x)
        else:
            gru_layer = GRU(
                hidden_units[i],
                return_sequences=ret_seq,
                kernel_regularizer=l2(decay),
                unroll=False,
            )
            if bidirectional:
                x = Bidirectional(gru_layer, name=name + f"_{i+1}")(x)
            else:
                gru_layer._name = name + f"_{i+1}"
                x = gru_layer(x)

        if layer_norm:
            x = LayerNormalization(name=name + f"_LN_{i+1}")(x)

    x = Reshape((1, -1), name=name + "_reshape")(x)
    x = GlobalAveragePooling1D(name=name + "_gap")(x)

    if regress:
        for i, units in enumerate(head_units):
            x = Dense(units, activation="relu", name=name + f"_dense{i+1}")(x)
            x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)
        x = Dense(1, activation="linear", name=name + "_output")(x)

    model_bigru = Model(inputs=signal_input_bigru, outputs=x, name=name)
    if regress:
        model_bigru.compile(
            loss=MeanSquaredError(),
            optimizer=get_optimizer(learning_r),
            metrics=[root_mse],
        )

    return model_bigru


def create_cnn(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    x_shape=[1, 1],
    name="CNN",
    filters=[32, 64, 128],
    kernel_size=[5, 5, 3],
    apply_dropout=[True, True, True],
    apply_pooling=[True, True, True],
    pooling=[2, 2, 2],
    scalogram_cwt=None,
    pooling_type="avg",
    regress=True,
    head_units=None,
    **kwargs,
):
    """
    Creates a 1D or 2D CNN model based on input shape.

    Args:
        learning_r (float): Learning rate.
        decay (float): Weight decay.
        dropout (float): Dropout rate.
        x_shape (tuple): Input shape.
        name (str): Model name.
        filters (list): List of filter counts for each convolutional block.
        kernel_size (list): List of kernel sizes for each convolutional block.
        apply_dropout (list): List of booleans indicating whether to apply dropout after each block.
        apply_pooling (list): List of booleans indicating whether to apply pooling after each block.
        scalogram_cwt (str): If not None, assumes 2D inputs from scalograms.
        pooling_type (str): 'max' or 'avg'. Defaults to 'max'.
        regress (bool): If True, add a regression head and compile the model.
            Set False to return a feature extractor only.
        head_units (list | None): Dense head sizes before output, e.g. [64, 32].
            None (default) uses [64, 32] for backward compatibility.

    Returns:
        keras.Model: Compiled Keras model.
    """
    is_2d = scalogram_cwt is not None
    if head_units is None:
        head_units = [64, 32]
    padding = "valid" if x_shape[2] == 1 else "same"
    signal_qty = x_shape[-1]

    def start_pooling(x, pool_size, name_p, is_2d_p):
        """Select and apply a 1-D or 2-D pooling layer based on pooling_type.

        Args:
            x: Keras tensor to pool.
            pool_size (int or tuple): Pool window size.
            name_p (str): Layer name.
            is_2d_p (bool): Apply 2-D pooling when True.

        Returns:
            Keras tensor after pooling.
        """
        if pooling_type == "max":
            if not is_2d_p:
                return MaxPooling1D(pool_size=pool_size, name=name_p)(x)
            else:
                return MaxPooling2D(pool_size=pool_size, name=name_p, padding=padding)(
                    x
                )
        else:
            if not is_2d_p:
                return AveragePooling1D(pool_size=pool_size, name=name_p)(x)
            else:
                return AveragePooling2D(
                    pool_size=pool_size, name=name_p, padding=padding
                )(x)

    input_bn = kwargs.get("input_bn", True)
    head_bn = kwargs.get("head_bn", True)

    signal_input_cnn = Input(x_shape[1:], name=name + "_signal_input")

    if not is_2d and input_bn:
        x = BatchNormalization(name=name + "_BN_input")(signal_input_cnn)
    else:
        x = signal_input_cnn if is_2d else signal_input_cnn

    x = conv_bn_relu(
        x if not is_2d else signal_input_cnn,
        filters[0],
        kernel_size[0],
        name=name,
        line=1,
        decay=decay,
        padding=padding,
        signal_qty=signal_qty,
        is_2d=is_2d,
        grouping=True,
    )
    if apply_pooling[0]:
        x = start_pooling(x, pooling[0], name + "_pool_1", is_2d)

    if apply_dropout[0]:
        x = Dropout(dropout, name=name + "_drop_1")(x)

    for i in range(1, len(filters)):
        x = conv_bn_relu(
            x,
            filters[i],
            kernel_size[i],
            name=name,
            line=i + 1,
            decay=decay,
            padding=padding,
            signal_qty=signal_qty,
            is_2d=is_2d,
        )
        if apply_pooling[i]:
            x = start_pooling(x, pooling[i], name + f"_pool_{i+1}", is_2d)

        if apply_dropout[i]:
            x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)

    if is_2d:
        x = GlobalAveragePooling2D(name=name + "_gap")(x)
    else:
        x = GlobalAveragePooling1D(name=name + "_gap")(x)

    if regress:
        for i, units in enumerate(head_units):
            x = Dense(units, name=name + f"_dense{i+1}")(x)
            if head_bn:
                x = BatchNormalization(name=name + f"_bn_dense{i+1}")(x)
            x = Activation("relu", name=name + f"_relu_dense{i+1}")(x)
            x = Dropout(dropout, name=name + f"_head_drop_{i+1}")(x)
        x = Dense(1, activation="linear", name=name + "_output")(x)

    model_cnn = Model(inputs=signal_input_cnn, outputs=x, name=name)
    if regress:
        model_cnn.compile(
            loss=MeanSquaredError(),
            optimizer=get_optimizer(learning_r),
            metrics=[root_mse],
        )

    return model_cnn


def create_cnn_lstm(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    x_shape=[1, 1],
    name="CNN_LSTM",
    filters=[32, 64, 128],
    kernel_size=[5, 5, 3],
    lstm_units=[64],
    scalogram_cwt=None,
    rnn_type="lstm",
    pooling_type="avg",
    attention=False,
    regress=True,
    bidirectional=True,
    head_units=None,
    **kwargs,
):
    """
    Creates a CRNN (CNN + RNN) model.
    Structure: Conv Blocks -> Recurrent Layer (BiLSTM/BiGRU) -> Dense Head.

    Args:
        learning_r (float): Learning rate.
        decay (float): L2 weight decay.
        dropout (float): Dropout rate.
        x_shape (list): Input shape (e.g. ``[N, T, C]``).
        name (str): Model name prefix. Defaults to ``'CNN_LSTM'``.
        filters (list[int]): Filter counts per CNN block.
        kernel_size (list[int]): Kernel sizes per CNN block.
        lstm_units (list[int]): Hidden unit counts per RNN layer.
        scalogram_cwt (str | None): If not None, activates 2-D mode for
            scalogram inputs.
        rnn_type (str): ``'lstm'`` or ``'bigru'`` / ``'gru'``.
        pooling_type (str): ``'avg'`` or ``'max'``.
        attention (bool): Add a CBAM attention block after the CNN stack.
        regress (bool): If True, add a regression head and compile the model.
        bidirectional (bool): Wrap each RNN layer in Bidirectional.
        head_units (list[int] | None): Dense head widths. Defaults to ``[64, 32]``.

    Returns:
        keras.Model: Compiled (when ``regress=True``) Keras model.
    """
    input_bn = kwargs.get("input_bn", True)
    layer_norm = kwargs.get("layer_norm", True)

    is_2d = scalogram_cwt is not None
    if head_units is None:
        head_units = [64, 32]

    padding = "valid"
    signal_qty = x_shape[-1]

    signal_input = Input(x_shape[1:], name=name + "_signal_input")
    x = signal_input
    if input_bn:
        x = BatchNormalization(name=name + "_BN_input")(x)

    for i in range(len(filters)):
        x = conv_bn_relu(
            x,
            filters[i],
            kernel_size[i] if not is_2d else (kernel_size[i], kernel_size[i]),
            name=name,
            line=i + 1,
            decay=decay,
            padding=padding,
            signal_qty=signal_qty,
            is_2d=is_2d,
        )
        
        def apply_pool(x, pool_size, name_p, is_2d_p):
            """Select and apply a 1-D or 2-D pooling layer based on pooling_type.

            Args:
                x: Keras tensor to pool.
                pool_size (int or tuple): Pool window size.
                name_p (str): Layer name.
                is_2d_p (bool): Apply 2-D pooling when True.

            Returns:
                Keras tensor after pooling.
            """
            if pooling_type == "max":
                if not is_2d_p:
                    return MaxPooling1D(pool_size=pool_size, name=name_p)(x)
                else:
                    return MaxPooling2D(
                        pool_size=pool_size, name=name_p, padding=padding
                    )(x)
            else:
                if not is_2d_p:
                    return AveragePooling1D(pool_size=pool_size, name=name_p)(x)
                else:
                    return AveragePooling2D(
                        pool_size=pool_size, name=name_p, padding=padding
                    )(x)

        if not is_2d:
            x = apply_pool(x, 2, name + f"_pool_{i+1}", is_2d)
        else:
            x = apply_pool(x, (2, 2), name + f"_pool_{i+1}", is_2d)

        if dropout > 0:
            x = Dropout(dropout, name=name + f"_drop_cnn_{i+1}")(x)

    if attention:
        x = cbam_block(x, is_2d=is_2d, name=name + "_cbam")

    if is_2d:
        x = TimeDistributed(Flatten(), name=name + "_flatten_freq")(x)

    is_torch = os.environ.get("KERAS_BACKEND") == "torch"
    for i in range(len(lstm_units)):
        return_seq = (
            i < len(lstm_units) - 1
        )

        if is_torch:
            if rnn_type == "bigru":
                rnn_layer = FastTorchGRU(
                    lstm_units[i],
                    return_sequences=return_seq,
                    bidirectional=bidirectional,
                    decay=decay,
                )
                name_suffix = f"gru_{i+1}"
            else:
                rnn_layer = FastTorchLSTM(
                    lstm_units[i],
                    return_sequences=return_seq,
                    bidirectional=bidirectional,
                    decay=decay,
                )
                name_suffix = f"lstm_{i+1}"

            x = rnn_layer(x)

        else:
            if rnn_type == "bigru":
                rnn_layer = GRU(
                    lstm_units[i],
                    return_sequences=return_seq,
                    kernel_regularizer=l2(decay),
                    unroll=False,
                )
                name_suffix = f"gru_{i+1}"
            else:
                rnn_layer = LSTM(
                    lstm_units[i],
                    return_sequences=return_seq,
                    kernel_regularizer=l2(decay),
                    unroll=False,
                )
                name_suffix = f"lstm_{i+1}"

            if bidirectional:
                x = Bidirectional(rnn_layer, name=name + f"_{name_suffix}")(x)
            else:
                rnn_layer._name = name + f"_{name_suffix}"
                x = rnn_layer(x)

        if layer_norm:
            x = LayerNormalization(name=name + f"_ln_{name_suffix}")(x)

    if regress:
        for i, units in enumerate(head_units):
            x = Dense(units, activation="relu", name=name + f"_dense{i+1}")(x)
            x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)
        x = Dense(1, activation="linear", name=name + "_output")(x)

    model = Model(inputs=signal_input, outputs=x, name=name)
    if regress:
        model.compile(
            loss=MeanSquaredError(),
            optimizer=get_optimizer(learning_r),
            metrics=[RootMeanSquaredError(name="root_mse")],
        )

    return model


def create_ensemble(
    learning_r=1e-3,
    decay=1e-5,
    dropout=0.2,
    xproc_shape=[1, 1],
    name="ens",
    hidden_units=[128, 64],
    model_list=[],
    proc_input=False,
    freeze_base=True,
    **kwargs,
):
    """
    Creates an ensemble model from a list of base models.
    Can also integrate process parameters via a separate MLP path.

    Args:
        learning_r (float): Learning rate.
        decay (float): L2 weight decay.
        dropout (float): Dropout rate.
        xproc_shape (list): Shape of the process-parameter input.
        name (str): Model name prefix. Defaults to ``'ens'``.
        hidden_units (list[int]): Dense head widths. Defaults to ``[128, 64]``.
        model_list (list): Base Keras models to ensemble; all are connected in
            parallel and their outputs are concatenated.
        proc_input (bool): Add a separate process-parameter MLP branch.
        freeze_base (bool): Freeze all layers of the base models.

    Returns:
        keras.Model: Compiled Keras ensemble model.
    """
    if len(model_list) == 0:
        raise ValueError("Ensemble requires at least one base model in 'model_list'.")

    if freeze_base:
        for i in range(len(model_list)):
            model = model_list[i]
            for layer in model.layers:
                layer.trainable = False

    if proc_input:
        process_input = Input(shape=(xproc_shape[1],), name="process_input")
        y = Dense(8, activation="relu", name=f"proc_dense")(process_input)
        y = BatchNormalization(name=f"proc_BN")(y)
        y = Dense(4, activation="linear", name="proc_output")(y)
        model_process_input = Model(
            inputs=process_input, outputs=y, name="process_data"
        )

    ensemble_inputs = [model.input for model in model_list]
    if proc_input:
        ensemble_inputs.append(model_process_input.input)
    ensemble_outputs = [model.layers[-1].output for model in model_list]
    if proc_input:
        ensemble_outputs.append(model_process_input.output)

    merge = concatenate(ensemble_outputs, name=name + "_concat")
    x = BatchNormalization(name=name + "_concat_BN")(merge)

    for i in range(len(hidden_units)):
        x = Dense(
            hidden_units[i],
            activation="relu",
            kernel_regularizer=l2(decay),
            name=name + f"_{i+1}",
        )(x)
        x = BatchNormalization(name=name + f"_BN_{i+1}")(x)

    if dropout != 0:
        x = Dropout(dropout, name=name + f"_drop_{i+1}")(x)

    x = Dense(1, activation="linear", name=name + "_output")(x)
    model_ens = Model(inputs=ensemble_inputs, outputs=x)

    model_ens.compile(
        loss=MeanSquaredError(),
        optimizer=get_optimizer(learning_r),
        metrics=[RootMeanSquaredError(name="root_mse")],
    )
    return model_ens


def create_proc_model(
    base_model_fn,
    base_model_params,
    ensemble_params,
    model_name=None,
    conditioning="proc",
    **kwargs,
):
    """
    Wraps a base model with conditioning inputs (process params, FiLM).

    Builds the base model with ``regress=False`` and taps its ``.output`` directly.

    Args:
        base_model_fn (callable): Factory function to create the base model.
        base_model_params (dict): Keyword arguments for ``base_model_fn``.
        ensemble_params (dict): Configuration for the conditioning path.
            Must include:
            - "xproc_shape": Tuple, shape of the processed input (1D).
            - "learning_r": Float, learning rate.
            - "hidden_units": List[int], dimensions for head dense layers.
              Defaults to [64, 32].
            - "dropout": Float, dropout rate.
        model_name (str, optional): Name for the wrapper model.
        conditioning (str): "proc" | "film".
            - "proc": process params.
            - "film": FiLM modulation from proc params on encoder features.
        **kwargs: Additional arguments.

    Returns:
        keras.Model: A compiled Keras model.
    """
    if base_model_fn is None:
        raise ValueError("base_model_fn is required.")

    base_params = {**base_model_params, "regress": False}
    base_model = base_model_fn(**base_params)
    signal_input = base_model.input
    base_output = base_model.output
    D = int(base_output.shape[-1])

    inputs = [signal_input] if not isinstance(signal_input, list) else list(signal_input)
    concat_parts = [base_output]

    xproc_shape = ensemble_params.get("xproc_shape", (14,))
    proc_dim = xproc_shape[-1] if len(xproc_shape) >= 1 else xproc_shape[0]
    dropout_rate = ensemble_params.get("dropout", 0.2)
    wrapper_name = model_name if model_name else ensemble_params.get("name", "ProcWrapper")

    proc_input = None

    if conditioning in ("proc", "film"):
        proc_input = Input(shape=(proc_dim,), name="proc_input")
        inputs.append(proc_input)

        if conditioning == "film":
            film_params = Dense(2 * D, use_bias=True, name="film_gen")(proc_input)
            gamma, beta = keras.ops.split(film_params, 2, axis=-1)
            base_output = keras.ops.add(
                keras.ops.multiply(gamma + 1.0, base_output), beta
            )
            concat_parts = [base_output]
        else:
            p = Dense(16, activation="relu", name="proc_d1")(proc_input)
            p = Dense(16, activation="relu", name="proc_d2")(p)
            concat_parts.append(p)

    if len(concat_parts) > 1:
        x = Concatenate(name="ens_concat")(concat_parts)
    else:
        x = concat_parts[0]

    hidden_units = ensemble_params.get("hidden_units", [64, 32])
    for i, units in enumerate(hidden_units):
        x = Dense(units, activation="relu", name=f"{wrapper_name}_ens_dense_{i}")(x)
        x = Dropout(dropout_rate, name=f"{wrapper_name}_ens_drop_{i}")(x)

    output = Dense(1, activation="linear", name=f"{wrapper_name}_output")(x)

    final_model = Model(inputs=inputs, outputs=output, name=wrapper_name)

    learning_rate = ensemble_params.get("learning_r", 1e-3)
    final_model.compile(
        loss=MeanSquaredError(),
        optimizer=get_optimizer(learning_rate),
        metrics=[RootMeanSquaredError(name="root_mse")],
    )

    return final_model

class FiLMLayer(keras.Layer):
    """Feature-wise Linear Modulation.

    Modulates a feature map x (batch, T, C) using scale γ and shift β computed
    from process parameters proc_params (batch, P).

    Intended placement: **inside** each Conv block, after BatchNorm and before
    the ReLU activation.  Pass ``use_film=True`` to the Conv block constructor
    to enable this path; backbone functions receive FiLM via ``add_film_*``
    wrappers (not via this layer directly at inference time).

    Args:
        num_channels (int): Number of feature channels C.
        name (str): Layer name prefix.
    """

    def __init__(self, num_channels: int, name: str = "film", **kwargs):
        """Initialise FiLMLayer.

        Args:
            num_channels (int): Number of feature channels C to modulate.
            name (str): Layer name prefix. Defaults to ``'film'``.
            **kwargs: Forwarded to ``keras.Layer.__init__``.
        """
        super().__init__(name=name, **kwargs)
        self.num_channels = num_channels
        self.film_gen = keras.layers.Dense(
            2 * num_channels, use_bias=True, name=name + "_gen"
        )

    def call(self, x, proc_params):
        """
        Args:
            x: (batch, T, C) — feature map after BN, before activation.
            proc_params: (batch, P) — process parameter vector.
        Returns:
            γ ⊙ x + β  with same shape as x.
        """
        film = self.film_gen(proc_params)
        gamma, beta = keras.ops.split(film, 2, axis=-1)
        gamma = keras.ops.expand_dims(gamma, axis=1)
        beta = keras.ops.expand_dims(beta, axis=1)
        return gamma * x + beta

    def get_config(self):
        """Return serialisable layer configuration dict."""
        cfg = super().get_config()
        cfg.update({"num_channels": self.num_channels})
        return cfg


def create_film_model(
    base_create_fn,
    base_params,
    proc_shape,
    model_name="Model_Film",
):
    """Late-fusion FiLM: builds a backbone with ``regress=False``, then applies
    Feature-wise Linear Modulation (γ ⊙ features + β) to the GAP output vector
    conditioned on process parameters before the regression head.

    This approach keeps the backbone architecture intact and injects
    process-parameter conditioning once at the feature level.  ``gamma`` is
    initialised near 1 (identity) so early training is not destabilised.

    Args:
        base_create_fn: Backbone creator, e.g. ``create_resnet``.
        base_params (dict): Keyword args for ``base_create_fn``; ``regress`` is
            forced to ``False`` here and must **not** be set in this dict.
        proc_shape (tuple): Shape of proc_params with or without the batch dim;
            only the last dimension is used, e.g. ``(N, P)`` or ``(P,)``.
        model_name (str): Name for the returned :class:`keras.Model`.

    Returns:
        Compiled :class:`keras.Model` with inputs ``[signal, proc]``.
    """
    backbone = base_create_fn(**{**base_params, "regress": False})
    signal_input = backbone.input
    features = backbone.output
    f_dim = features.shape[-1]

    proc_input = keras.Input(shape=(proc_shape[-1],), name="film_proc_input")

    film_params = keras.layers.Dense(
        2 * f_dim, use_bias=True, name="film_gen"
    )(proc_input)
    gamma, beta = keras.ops.split(film_params, 2, axis=-1)
    modulated = keras.ops.add(keras.ops.multiply(gamma + 1.0, features), beta)

    all_inputs = [signal_input, proc_input]
    x = modulated
    x = keras.layers.Dense(64, activation="relu", name="film_dense1")(x)
    x = keras.layers.Dropout(base_params.get("dropout", 0.2), name="film_drop")(x)
    out = keras.layers.Dense(1, activation="linear", name="film_output")(x)

    model = keras.Model(inputs=all_inputs, outputs=out, name=model_name)
    model.compile(
        loss=MeanSquaredError(),
        optimizer=get_optimizer(base_params.get("learning_r", 1e-3)),
        metrics=[RootMeanSquaredError(name="root_mse")],
    )
    return model


def get_model(
    model_name,
    input_shape,
    hps,
    proc_shape=None,
    pretrained_baselines=None,
    **kwargs,
):
    """
    Factory function to create compiled Keras models for the DeepTCM benchmark.

    This function acts as the central dispatcher. It:
    1.  Parses the `model_name` to determine the architecture (e.g., CNN, ResNet).
    2.  Detects if a `_Proc` or `_Film` suffix is requested (e.g., `ResNet_Proc`,
        `ResNet_Film`, `ResNet_Film_Stats`).
    3.  Configures the model based on hyperparameters (`hps`) and input constraints.
    4.  Wraps the model in `create_proc_model` if `_Proc` is requested,
        or `create_film_model` if `_Film` is requested (unless Ensemble).

    Args:
        model_name (str): Architecture name (``"CNN"``, ``"ResNet"``, etc.).
        input_shape (tuple): Signal input shape, e.g. ``(N_windows, L, C)``.
        hps (dict): Hyperparameters.
        proc_shape (tuple, optional): Shape of processed data; required for ``_Proc`` variants.
        pretrained_baselines (dict, optional): Baseline paths for Ensemble.
        **kwargs: Additional arguments passed through (e.g. ``original_name``).

    Returns:
        keras.Model: A compiled Keras model ready for training.

    Raises:
        ValueError: If an unknown model is requested or a strict baseline is requested with `_Proc`.
    """
    use_proc = False
    original_name = model_name

    if model_name.endswith("_Proc") and "Ensemble" not in model_name:
        use_proc = True
        model_name = model_name.replace("_Proc", "")

        if model_name in ["CNN"]:
            raise ValueError(
                f"Model '{original_name}' is invalid. '{model_name}' is a strict baseline "
                "and does not support the '_Proc' variant."
            )

    use_film = False
    if "_Film" in model_name and "Ensemble" not in model_name:
        use_film = True
        model_name = model_name.replace("_Film", "")

    if use_proc and use_film:
        raise ValueError(
            f"Model '{original_name}' combines '_Proc' and '_Film' suffixes, which are "
            "mutually exclusive.  Use '_Film' for FiLM conditioning or '_Proc' for "
            "concatenative process conditioning."
        )

    scalogram_mode = hps.get("scalogram") if hps.get("scalogram") != "none" else None

    params = {
        "learning_r": hps.get("learning_rate", 1e-3),
        "decay": hps.get("weight_decay", 1e-5),
        "dropout": hps.get("dropout", 0.2),
        "x_shape": input_shape,
        "name": kwargs.get("name", model_name),
        "scalogram_cwt": scalogram_mode,
    }

    create_fn = None

    if model_name == "CNN":
        create_fn = create_cnn
        base   = hps.get("filters_base") or 8
        _n_cnn = hps.get("cnn_layers") or 2
        _cnn_kernels_tmpl = [5, 5, 3, 3, 3]
        _cnn_kernels = (_cnn_kernels_tmpl + [3] * max(0, _n_cnn - len(_cnn_kernels_tmpl)))[:_n_cnn]
        params["filters"]       = [base * (2 ** i) for i in range(_n_cnn)]
        params["kernel_size"]   = _cnn_kernels
        params["apply_dropout"] = [True] * _n_cnn
        params["apply_pooling"] = [True] * _n_cnn
        params["pooling"]       = [2] * _n_cnn
        params["pooling_type"]  = hps.get("pooling", "avg")

    elif model_name == "LSTM":
        create_fn = create_lstm
        _n_layers = hps.get("lstm_layers", 2)
        params["hidden_units"] = [hps.get("lstm_units", 64)] * _n_layers

    elif model_name == "BiGRU":
        create_fn = create_bigru
        _n_layers = hps.get("lstm_layers", 2)
        params["hidden_units"] = [hps.get("lstm_units", 64)] * _n_layers
        
    elif "CNN_LSTM" in model_name:
        create_fn = create_cnn_lstm
        base = hps.get("filters_base") or 8
        _n_cnn = hps.get("cnn_layers") or 2
        _cnn_kernels_tmpl = [5, 5, 3, 3, 3]
        _cnn_kernels = (_cnn_kernels_tmpl + [3] * max(0, _n_cnn - len(_cnn_kernels_tmpl)))[:_n_cnn]
        params["filters"] = [base * (2**i) for i in range(_n_cnn)]
        params["kernel_size"] = _cnn_kernels
        _n_lstm = hps.get("lstm_layers", 1)
        params["lstm_units"] = [hps.get("lstm_units", 64)] * _n_lstm
        params["rnn_type"] = hps.get("rnn_type", "lstm")
        params["attention"] = model_name == "RobustCNN_LSTM"
        params["pooling_type"] = hps.get("pooling", "avg")

    elif "ResNet" in model_name:
        create_fn = create_resnet
        _res_base = hps.get("filters_base") or 8
        _res_n    = hps.get("cnn_layers") or 2
        _res_kernels_tmpl = [5, 5, 3, 3, 3]
        _res_kernels = (_res_kernels_tmpl + [3] * max(0, _res_n - len(_res_kernels_tmpl)))[:_res_n]
        params["filters"]       = [_res_base * (2 ** i) for i in range(_res_n)]
        params["kernel_size"]   = _res_kernels
        params["apply_dropout"] = [True] * _res_n
        params["apply_pooling"] = [True] * _res_n
        params["is_resnet"]     = [False] + [True] * max(0, _res_n - 1)
        params["pooling_type"]  = hps.get("pooling", "avg")
        if model_name == "RobustResNet":
            params["attention"] = True
            params["name"] = "RobustResNet"

    elif "Ensemble" in model_name:
        if not pretrained_baselines:
            raise ValueError("Ensemble requires 'pretrained_baselines' dictionary.")

        model_list = []
        for base_name in ["CNN", "LSTM", "BiGRU"]:
            path = pretrained_baselines.get(base_name)
            if not path:
                print(
                    f"Warning: Baseline '{base_name}' path not found in `pretrained_baselines`."
                )
                continue

            try:
                m = keras.saving.load_model(
                    path,
                    custom_objects={
                        "root_mse": RootMeanSquaredError(name="root_mse"),
                        "FastTorchLSTM": FastTorchLSTM,
                        "FastTorchGRU": FastTorchGRU,
                        "Custom>FastTorchLSTM": FastTorchLSTM,
                        "Custom>FastTorchGRU": FastTorchGRU,
                    },
                )

                if hasattr(m, "input") and m.input is not None:
                    if m.input.shape[0] is None:
                        dummy_shape = (1, *m.input.shape[1:])
                    else:
                        dummy_shape = m.input.shape

                    _ = m(ops.zeros(dummy_shape, dtype=m.compute_dtype))
                    logger.info(f"Built baseline {base_name} with shape {dummy_shape}")

                m._name = f"{base_name}_Loaded"
                m.trainable = False
                model_list.append(m)
            except Exception as e:
                logger.warning(f"Could not load baseline {base_name}: {e}")

        create_fn = create_ensemble
        params["model_list"] = model_list

        is_proc_ens = "Proc" in original_name
        params["proc_input"] = is_proc_ens
        if is_proc_ens:
            params["xproc_shape"] = proc_shape

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if create_fn is None:
        raise ValueError(f"Create function not resolved for: {model_name}")

    if "Ensemble" in model_name:
        return create_fn(**params)

    if use_film:
        if proc_shape is None:
            raise ValueError(
                f"get_model(): proc_shape must be provided for '{original_name}'"
            )
        return create_proc_model(
            base_model_fn=create_fn,
            base_model_params=params,
            ensemble_params={
                "xproc_shape": proc_shape,
                "learning_r": params["learning_r"],
                "dropout": params["dropout"],
                "hidden_units": [64],
            },
            model_name=original_name,
            conditioning="film",
        )
    if use_proc:
        _eff_conditioning = hps.get("conditioning", "proc")
        if _eff_conditioning not in ("proc", "film"):
            _eff_conditioning = "proc"
        if _eff_conditioning == "film":
            _head_units = [64]
        else:
            _head_units = [16]
            
        return create_proc_model(
            base_model_fn=create_fn,
            base_model_params=params,
            ensemble_params={
                "xproc_shape": proc_shape,
                "learning_r": params["learning_r"],
                "dropout": params["dropout"],
                "hidden_units": _head_units,
            },
            model_name=original_name,
            conditioning=_eff_conditioning,
        )
    
    return create_fn(**params)
