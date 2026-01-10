import keras_tuner as kt
import autokeras as ak
import keras
from keras import layers
import keras.ops as ops
from keras.metrics import RootMeanSquaredError

import tree

from typing import Optional
from typing import Union

from autokeras import adapters, analysers, preprocessors, keras_layers
from autokeras import hyper_preprocessors as hpps_module
from autokeras.blocks import reduction, preprocessing
from autokeras.engine import head as head_module
from keras_tuner.engine import hyperparameters
from autokeras.utils import types, io_utils, utils

from models.dl_models import Patches, PatchEncoder, vit_transformer_block, ts_transformer_block

class ReshapeBlock(ak.Block):
    def __init__(self, target_shape, **kwargs):
        super().__init__(**kwargs)
        self.target_shape = target_shape

    def get_config(self):
        config = super().get_config()
        config.update({"target_shape": self.target_shape})
        return config

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = layers.Reshape(self.target_shape)(input_node)
        return output_node

class RNNBlock(ak.Block):
    """
    A custom AutoKeras Block that implements an RNN layer (GRU or LSTM) with 
    optimisable number of units.
    """
    def __init__(self, 
                return_sequences: bool = False,
                bidirectional: Optional[Union[bool, hyperparameters.Boolean]] = None,
                num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
                layer_type: Optional[Union[str, hyperparameters.Choice]] = None,
                units: Optional[Union[int, hyperparameters.Choice]] = None,
                **kwargs):
        super().__init__(**kwargs)
        
        self.return_sequences = return_sequences
        self.bidirectional = utils.get_hyperparameter(
            bidirectional,
            hyperparameters.Boolean("bidirectional", default=True),
            bool,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", [1, 2, 3], default=2),
            int,
        )
        self.layer_type = utils.get_hyperparameter(
            layer_type,
            hyperparameters.Choice(
                "layer_type", ["gru", "lstm"], default="gru"
            ),
            str,
        )
        self.units = utils.get_hyperparameter(
            units,
            hyperparameters.Choice("units", [32, 128, 256], default=32),
            int,
        )       

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "return_sequences": self.return_sequences,
                "bidirectional": io_utils.serialize_block_arg(
                    self.bidirectional
                ),
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "layer_type": io_utils.serialize_block_arg(self.layer_type),
                "units": io_utils.serialize_block_arg(self.units),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["bidirectional"] = io_utils.deserialize_block_arg(
            config["bidirectional"]
        )
        config["num_layers"] = io_utils.deserialize_block_arg(
            config["num_layers"]
        )
        config["layer_type"] = io_utils.deserialize_block_arg(
            config["layer_type"]
        )
        config["units"] = io_utils.deserialize_block_arg(
            config["units"]
        )
        return cls(**config)

    def build(self, hp, inputs=None):
        """
        Builds the RNN block.

        Args:
            hp (kt.HyperParameters): Hyperparameter state.
            inputs (ak.Node): Input node to the block.

        Returns:
            ak.Node: The output node of the RNN block.
        """
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        
        # Hyperparameters
        units = utils.add_to_hp(self.units, hp)
        layer_type = utils.add_to_hp(self.layer_type, hp)
        bidirectional = utils.add_to_hp(self.bidirectional, hp)
        num_layers = utils.add_to_hp(self.num_layers, hp)
        rnn_layers = {"gru": layers.GRU, "lstm": layers.LSTM}
        in_layer = rnn_layers[layer_type]
        
        output_node = input_node
        for i in range(num_layers):
            return_sequences = True
            if i == num_layers - 1:
                return_sequences = self.return_sequences
            if bidirectional:
                output_node = layers.Bidirectional(  # pragma: no cover
                    in_layer(units, return_sequences=return_sequences)
                )(output_node)
            else:
                output_node = in_layer(
                    units, return_sequences=return_sequences
                )(output_node)
        return output_node

class SignalTransformerBlock(ak.Block):
    """
    A custom AutoKeras Block that implements a Transformer Encoder for signals.
    Handles both 1D signals and 2D scalograms.
    """
    def __init__(self,
                 num_heads: Optional[Union[int, hyperparameters.Choice]] = None,
                 latent_dim: Optional[Union[int, hyperparameters.Choice]] = None,
                 dense_dim: Optional[Union[int, hyperparameters.Choice]] = None,
                 dropout: Optional[Union[float, hyperparameters.Choice]] = None,
                 num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
                 patch_size: Optional[Union[int, hyperparameters.Choice]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = utils.get_hyperparameter(
            num_heads, hyperparameters.Choice("num_heads", [4, 8], default=4), int
        )
        self.latent_dim = utils.get_hyperparameter(
            latent_dim, hyperparameters.Choice("latent_dim", [64, 128, 256], default=64), int
        )
        self.dense_dim = utils.get_hyperparameter(
            dense_dim, hyperparameters.Choice("dense_dim", [64, 128, 256], default=64), int
        )
        self.dropout = utils.get_hyperparameter(
            dropout, hyperparameters.Choice("dropout", [0.0, 0.25, 0.5], default=0.0), float
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers, hyperparameters.Choice("num_layers", [2, 4, 8], default=2), int
        )
        self.patch_size = utils.get_hyperparameter(
            patch_size, hyperparameters.Choice("patch_size", [8, 16], default=8), int
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": io_utils.serialize_block_arg(self.num_heads),
                "latent_dim": io_utils.serialize_block_arg(self.latent_dim),
                "dense_dim": io_utils.serialize_block_arg(self.dense_dim),
                "dropout": io_utils.serialize_block_arg(self.dropout),
                "num_layers": io_utils.serialize_block_arg(self.num_layers),
                "patch_size": io_utils.serialize_block_arg(self.patch_size),
            }
        )
        return config

    def build(self, hp, inputs=None):
        """
        Builds the Transformer block.
        """
        input_node = inputs
        shape = input_node.shape

        # Handle 1D signals reshaped by ImageInput (Batch, Time, 1, Channels)
        # Squeeze the width dimension if it is 1 to get back to (Batch, Time, Channels)
        if len(shape) == 4:
            if shape[2] == 1:
                input_node = ops.squeeze(input_node, axis=2)
            elif shape[3] == 1:
                input_node = ops.squeeze(input_node, axis=3)
            shape = input_node.shape

        # Handle 4D input (N, S, L, C) -> Scalogram
        is_2d = len(shape) == 4
        
        # Transformer Hyperparameters
        num_heads = utils.add_to_hp(self.num_heads, hp)
        latent_dim = utils.add_to_hp(self.latent_dim, hp)
        dense_dim = utils.add_to_hp(self.dense_dim, hp)
        dropout_rate = utils.add_to_hp(self.dropout, hp)
        num_layers = utils.add_to_hp(self.num_layers, hp)
        
        output_node = input_node

        if is_2d:
             # Vision Transformer Logic
            # x_shape is (Batch, H, W, C)
            _, height, width, channels = shape
            patch_size = utils.add_to_hp(self.patch_size, hp)

            output_node = Patches(patch_size)(output_node)
            # patches shape: (B, num_patches, patch_dims)
            
            # Calculate num_patches
            # Dynamic calculation not supported easily in Keras layer call for static shape ref in init
            # But PatchEncoder needs num_patches.
            # Assuming fixed size for search or calculating dynamically? 
            # dl_models implementation relies on static num_patches passed to PatchEncoder.
            num_patches = (height // patch_size) * (width // patch_size)
            
            # Encode patches
            output_node = PatchEncoder(num_patches, latent_dim)(output_node)
            
            # ViT Blocks
            output_node = vit_transformer_block(output_node, num_layers, latent_dim, num_heads, dense_dim=latent_dim*2, dropout=dropout_rate, name='ak_vit')
            
            # Output Head (ViT style: LN -> Flatten -> Dense -> Dropout...)
            # We return the feature representation before the final head for AutoKeras to handle?
            # Or we reduce it here. AutoKeras blocks usually return features to be flattened/processed by head.
            # But generic block should output something compatible with next blocks (usually dense or flattened)
            
            output_node = layers.LayerNormalization(epsilon=1e-6)(output_node)
            output_node = layers.Flatten()(output_node)
            # Provide a dense projection to ensure consistent output size independent of patch count
            output_node = layers.Dense(dense_dim)(output_node)
        else:
            # Time-Series Transformer Logic
            # TS Blocks
            output_node = ts_transformer_block(output_node, num_layers, head_size=latent_dim, num_heads=num_heads, ff_dim=dense_dim, dropout=dropout_rate, name='ak_ts')
            
            # Output Head (TS style: GAP)
            output_node = layers.GlobalAveragePooling1D(data_format="channels_last")(output_node)
            
        return output_node

class SignalBlock(ak.Block):
    """
    Block for signal data (based heavily on AutoKeras ImageBlock).
    It selects between ResNet, Xception, EfficientNet, and Transformer, 
    which is controlled by a hyperparameter, 'block_type'.

    # Arguments
        block_type: String. 'resnet', 'xception', 'transformer'. The type of Block
            to use. If unspecified, it will be tuned automatically.
        normalize: Boolean. Whether to channel-wise normalize the images.
            If unspecified, it will be tuned automatically.
        augment: Boolean. Whether to do image augmentation. If unspecified,
            it will be tuned automatically.
    """
    def __init__(
        self,
        block_type: Optional[Union[str, hyperparameters.Choice]] = None,
        normalize: Optional[Union[bool, hyperparameters.Boolean]] = None,
        augment: Optional[Union[bool, hyperparameters.Boolean]] = None,
        use_lstm = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_type = utils.get_hyperparameter(
            block_type,
            hyperparameters.Choice(
                "block_type", ["vanilla", "xception", "transformer"], default="transformer"
            ),
            str,
        )
        self.normalize = utils.get_hyperparameter(
            normalize,
            hyperparameters.Boolean("normalize", default=False),
            bool,
        )
        self.augment = utils.get_hyperparameter(
            augment,
            hyperparameters.Boolean("augment", default=False),
            bool,
        )
        self.use_lstm = utils.get_hyperparameter(
            use_lstm,
            hyperparameters.Boolean("use_lstm", default=False),
            bool,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "block_type": io_utils.serialize_block_arg(self.block_type),
                "normalize": io_utils.serialize_block_arg(self.normalize),
                "augment": io_utils.serialize_block_arg(self.augment),
            }
        )
        return config

    def _build_block(self, hp, output_node, block_type):
        if block_type == "resnet":
            block = ak.ResNetBlock(version="v2", pretrained=False)
        elif block_type == "xception":
            block = ak.XceptionBlock(pretrained=False)
        elif block_type == "transformer":
            block = SignalTransformerBlock()
        elif block_type == "vanilla":
            return ak.ConvBlock().build(hp, output_node)
        
        return block.build(hp, output_node)

    def build(self, hp, inputs=None):
        
        input_node = tree.flatten(inputs)[0]
        output_node = input_node

        normalize = utils.add_to_hp(self.normalize, hp)
        if normalize:
            with hp.conditional_scope("normalize", [True]):
                 output_node = preprocessing.Normalization().build(hp, output_node)

        augment = utils.add_to_hp(self.augment, hp)
        if augment:
             with hp.conditional_scope("augment", [True]):
                output_node = preprocessing.ImageAugmentation().build(hp, output_node)

        block_type = utils.add_to_hp(self.block_type, hp)
        with hp.conditional_scope("block_type", [block_type]):
             output_node = self._build_block(hp, output_node, block_type)

        if block_type != "transformer":
            use_lstm = utils.add_to_hp(self.use_lstm, hp)
            with hp.conditional_scope("use_lstm", [True]):
                # GlobalPooling for CNNs to get (B, C)
                output_node = layers.GlobalAveragePooling2D(data_format="channels_last")(output_node)
                output_node = layers.Dense(32, activation='relu')(output_node)
                
                # Reshape to (B, 1, C) for RNN.
                # We use (1, C) instead of (C, 1) because:
                # 1. Performance: Processing 2048+ channels as time steps is extremely slow and hard to learn.
                # 2. Semantics: Channels are concurrent features, not a temporal sequence.
                output_node = layers.Reshape((1, -1))(output_node)
                
                # Process then with RNN
                output_node = RNNBlock(num_layers=2, units=32, 
                                bidirectional=True, layer_type='gru').build(hp, output_node)

        return output_node

class RegressionHead(head_module.Head):
    """Regression Dense layers (copy of AutoKeras RegressionHead forcing output dims).

    The targets passing to the head would have to be np.ndarray. It can be
    single-column or multi-column. The values should all be numerical.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will be inferred from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `mean_squared_error`.
        metrics: A list of Keras metrics. Defaults to use `mean_squared_error`.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        loss: types.LossType = "mean_squared_error",
        metrics: Optional[types.MetricsType] = None,
        dropout: Optional[float] = None,
        **kwargs
    ):
        if metrics is None:
            metrics = [RootMeanSquaredError()]
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        self.output_dim = output_dim
        self.dropout = dropout

    def get_config(self):
        config = super().get_config()
        config.update({"output_dim": self.output_dim, "dropout": self.dropout})
        return config

    def build(self, hp, inputs=None):
        inputs = tree.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        output_node = reduction.Flatten().build(hp, output_node)
        output_node = layers.Dense(self.output_dim or self.shape[-1], name=self.name)(output_node)
        return output_node

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self._add_one_dimension = len(analyser.shape) == 1

    def get_adapter(self):
        return adapters.RegressionAdapter(name=self.name)

    def get_analyser(self):
        return analysers.RegressionAnalyser(
            name=self.name, output_dim=self.output_dim
        )

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []
        if self._add_one_dimension:
            hyper_preprocessors.append(  # pragma: no cover
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.AddOneDimension()
                )
            )
        return hyper_preprocessors

class AutoKerasHyperModel(kt.HyperModel):
    """
    A HyperModel that wraps an AutoKeras search within a Keras Tuner search.
    This allows for nested optimization of preprocessing and model architecture.
    """
    def __init__(self, config):
        """
        Initializes the hypermodel.

        Args:
            config (dict): Complete configuration dictionary.
        """
        self.config = config
        self.ak_config = config.get("autokeras", {})
        self.is_multimodal = True
        self.input_shape = None
        
    def set_input_shape(self, shape):
        """Sets the input shape for the model."""
        self.input_shape = shape
        
    def set_multimodal(self, status):
        """Sets the multi-modality status."""
        self.is_multimodal = status

    def set_project_name(self, name):
        """Sets the project name for the AutoKeras search."""
        self.project_name = name
        
    def build(self, hp=None):
        """
        Builds the AutoKeras model graph.

        Args:
            hp (kt.HyperParameters): Preprocessing hyperparameters.

        Returns:
            ak.AutoModel: The AutoKeras AutoModel instance.
        """
        # input_node will be a dictionary if multimodal or a single node
        if self.is_multimodal:
            input_signal = ak.ImageInput(name="signal")
            input_info = ak.Input(name="info")
        else:
            input_signal = ak.ImageInput()
            
        # Unified SignalBlock
        signal_out = SignalBlock(normalize=False, augment=False)(input_signal)
            
        if self.is_multimodal:
            # Info branch: Fixed DenseBlock per user spec
            info_out = ak.DenseBlock(
                num_layers=2, 
                num_units=32, 
                use_batchnorm=True, 
                dropout=0.25
            )(input_info)
            
            # Merge
            merged = ak.Merge(merge_type='concatenate')([signal_out, info_out])
            
            # Post-merge Head: Fixed DenseBlock per user spec
            output_node = ak.DenseBlock(
                num_layers=2, 
                num_units=64, 
                use_batchnorm=True, 
                dropout=0.25
            )(merged)
            
            inputs = [input_signal, input_info]
        else:
            output_node = signal_out
            inputs = input_signal
            
        output_node = RegressionHead(output_dim=1)(output_node)
            
        return ak.AutoModel(
            inputs=inputs, 
            outputs=output_node, 
            max_trials=self.ak_config.get("max_trials", 3),
            objective=self.ak_config.get("objective", "val_loss"),
            overwrite=self.ak_config.get("overwrite", False),
            directory=self.config.get("directory", "ak_project"),
            project_name=getattr(self, 'project_name', f"{self.config.get('project_name', 'ak_project')}"),
            tuner='bayesian'
        )