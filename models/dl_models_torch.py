################################################################################
# Copyright (c) 2026 José Joaquín Peralta Abadía.                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 06-06-2026                                                             #
# Author(s): José Joaquín Peralta Abadía                                       #
# E-mail: josejoaquin.peralta.abadia@gmail.com                                 #
################################################################################

"""
PyTorch Model Zoo for DeepTCM — Continual Learning compatible.

Mirrors the Keras model zoo in dl_models.py but produces pure torch.nn.Module
objects that Avalanche can consume directly.

Input convention:
    forward(x)  where  x = {'x': signal_tensor, 'proc_data': proc_tensor}
    - signal_tensor: (B, T, C) for 1D  or  (B, T, F, C) for 2D scalogram
    - proc_tensor:   (B, P)

Output: (B, 1) — VB regression value.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnRelu(nn.Module):
    """Conv → BatchNorm → ReLU block (1D or 2D)."""

    def __init__(self, in_channels, out_channels, kernel_size, padding="same",
                 is_2d=False, activation=True):
        """Initialise the CNN block layers (conv, BN, ReLU).

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels (filters).
            kernel_size (int or tuple): Convolution kernel size.
            padding (str): ``'same'`` (zero-pad to preserve length) or ``'valid'``.
            is_2d (bool): Use Conv2d/BN2d when True; Conv1d/BN1d otherwise.
            activation (bool): Apply ReLU after BN when True.
        """
        super().__init__()
        Conv = nn.Conv2d if is_2d else nn.Conv1d
        BN = nn.BatchNorm2d if is_2d else nn.BatchNorm1d

        if padding == "same":
            if isinstance(kernel_size, tuple):
                pad = tuple(k // 2 for k in kernel_size)
            else:
                pad = kernel_size // 2
        else:
            pad = 0

        self.conv = Conv(in_channels, out_channels, kernel_size, padding=pad)
        self.bn = BN(out_channels)
        self.activation = activation

    def forward(self, x):
        """Forward pass through the CNN block."""
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = F.relu(x)
        return x


class FeatureDenoising(nn.Module):
    """Denoising block: Conv(k=5) → BN → ReLU → Dropout."""

    def __init__(self, in_channels, out_channels, dropout=0.2, is_2d=False):
        """Initialise FeatureDenoising.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dropout (float): Dropout probability applied after ReLU.
            is_2d (bool): Use 2-D conv/BN when True.
        """
        super().__init__()
        Conv = nn.Conv2d if is_2d else nn.Conv1d
        BN = nn.BatchNorm2d if is_2d else nn.BatchNorm1d
        kernel = (5, 1) if is_2d else 5

        self.conv = Conv(in_channels, out_channels, kernel)
        self.bn = BN(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass through the ResNet block (with optional skip projection)."""
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.drop(x)
        return x


class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention).
      - Channel: GAP+GMP → shared MLP → sigmoid
      - Spatial:  channel avg+max → Conv(k=7) → sigmoid
    """

    def __init__(self, channels, ratio=8, is_2d=False):
        """Initialise CBAMBlock.

        Args:
            channels (int): Number of feature map channels.
            ratio (int): Channel-reduction ratio for the MLP bottleneck.
            is_2d (bool): Use 2-D spatial pooling/convolution when True.
        """
        super().__init__()
        self.is_2d = is_2d
        bottleneck = max(channels // ratio, 4)

        self.ca_fc1 = nn.Linear(channels, bottleneck)
        self.ca_fc2 = nn.Linear(bottleneck, channels)

        Conv = nn.Conv2d if is_2d else nn.Conv1d
        self.sa_conv = Conv(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        """Forward pass through the CNN extractor."""
        if self.is_2d:
            avg_pool = x.mean(dim=(2, 3))
            max_pool = x.amax(dim=(2, 3))
        else:
            avg_pool = x.mean(dim=2)
            max_pool = x.amax(dim=2)

        avg_out = self.ca_fc2(F.relu(self.ca_fc1(avg_pool)))
        max_out = self.ca_fc2(F.relu(self.ca_fc1(max_pool)))
        ca = torch.sigmoid(avg_out + max_out)

        if self.is_2d:
            ca = ca.unsqueeze(-1).unsqueeze(-1)
        else:
            ca = ca.unsqueeze(-1)
        x = x * ca

        if self.is_2d:
            sa_avg = x.mean(dim=1, keepdim=True)
            sa_max = x.amax(dim=1, keepdim=True)
        else:
            sa_avg = x.mean(dim=1, keepdim=True)
            sa_max = x.amax(dim=1, keepdim=True)

        sa = torch.sigmoid(self.sa_conv(torch.cat([sa_avg, sa_max], dim=1)))
        x = x * sa
        return x


class ResNetBlock(nn.Module):
    """Two-conv residual block with optional CBAM and auto-projection."""

    def __init__(self, in_channels, out_channels, kernel_size, attention=False,
                 is_2d=False):
        """Initialise ResNetBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output (residual) channels.
            kernel_size (int): Convolution kernel size for both conv layers.
            attention (bool): Add a CBAM attention module after the second conv.
            is_2d (bool): Use 2-D operations when True.
        """
        super().__init__()
        self.conv1 = ConvBnRelu(in_channels, out_channels, kernel_size,
                                padding="same", is_2d=is_2d, activation=True)
        self.conv2 = ConvBnRelu(out_channels, out_channels, kernel_size,
                                padding="same", is_2d=is_2d, activation=False)

        self.proj = None
        if in_channels != out_channels:
            Conv = nn.Conv2d if is_2d else nn.Conv1d
            self.proj = Conv(in_channels, out_channels, kernel_size=1)

        self.attention = CBAMBlock(out_channels, is_2d=is_2d) if attention else None

    def forward(self, x):
        """Forward pass through the ResNet extractor."""
        shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.attention is not None:
            out = self.attention(out)

        if self.proj is not None:
            shortcut = self.proj(shortcut)

        out = F.relu(out + shortcut)
        return out


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: γ⊙x + β.

    γ is initialised near 1 (identity) via the +1 offset so the backbone
    output is not disrupted early in training.
    """

    def __init__(self, num_features, proc_dim):
        """Initialise FiLMLayer.

        Args:
            num_features (int): Dimensionality D of the feature vector to modulate.
            proc_dim (int): Dimensionality P of the process-parameter input vector.
        """
        super().__init__()
        self.film_gen = nn.Linear(proc_dim, 2 * num_features)

    def forward(self, x, proc_params):
        """
        Args:
            x: (B, D) — encoder features.
            proc_params: (B, P) — process parameter vector.
        Returns:
            (γ+1)⊙x + β, same shape as x.
        """
        film = self.film_gen(proc_params)  # (B, 2D)
        gamma, beta = film.chunk(2, dim=-1)
        return (gamma + 1.0) * x + beta


class ProcEncoder(nn.Module):
    """Process parameter encoder.
     
      Layout: Linear → ReLU → Linear → ReLU.
    """

    def __init__(self, proc_dim, hidden=16):
        """Initialise ProcEncoder.

        Args:
            proc_dim (int): Dimensionality P of the input process-parameter vector.
            hidden (int): Width of the two hidden linear layers. Defaults to 16.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proc_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, x):
        """Apply FiLM: compute gamma/beta from proc_params, scale and shift features."""
        return self.net(x)

class TorchCNN(nn.Module):
    """1D/2D CNN matching Keras create_cnn().

    Layout: [ConvBnRelu → Pool → Dropout] × N → GAP → DenseHead.
    """

    def __init__(self, in_channels, filters, kernel_sizes, dropout=0.2,
                 pooling_type="avg", is_2d=False, head_units=None,
                 input_bn=True):
        """Initialise TorchCNN.

        Args:
            in_channels (int): Number of input channels.
            filters (list[int]): Filter counts for each ConvBnRelu block.
            kernel_sizes (list[int]): Kernel sizes for each block.
            dropout (float): Dropout probability applied after each pool.
            pooling_type (str): ``'avg'`` or ``'max'`` pooling after each block.
            is_2d (bool): Use 2-D conv/pool layers when True.
            head_units (list[int] | None): Dense head layer widths before the
                scalar output. Defaults to ``[64, 32]``.
            input_bn (bool): Apply BatchNorm to the input before the first block.
        """
        super().__init__()
        self.is_2d = is_2d
        self.input_bn = None
        if input_bn and not is_2d:
            self.input_bn = nn.BatchNorm1d(in_channels)

        layers = []
        ch = in_channels
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            layers.append(ConvBnRelu(ch, f, k, padding="same", is_2d=is_2d))
            Pool = (nn.AvgPool2d if is_2d else nn.AvgPool1d) if pooling_type == "avg" \
                else (nn.MaxPool2d if is_2d else nn.MaxPool1d)
            layers.append(Pool(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            ch = f
        self.features = nn.Sequential(*layers)

        if head_units is None:
            head_units = [64, 32]
        head = []
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.BatchNorm1d(hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x):
        """Return features before the regression head."""
        if self.input_bn is not None:
            x = self.input_bn(x)
        x = self.features(x)
        x = x.mean(dim=-1) if not self.is_2d else x.mean(dim=(-2, -1))
        x = self.head(x)
        return x

    def forward(self, x):
        """Forward pass: extract signal features, optionally apply FiLM, output VB prediction."""
        x = self.encode(x)
        return self.output(x)


class TorchResNet(nn.Module):
    """ResNet matching Keras create_resnet().

    Layout: FeatureDenoising → [Plain / ResNetBlock] × N → GAP → DenseHead.
    """

    def __init__(self, in_channels, filters, kernel_sizes, dropout=0.2,
                 pooling_type="avg", attention=False, is_2d=False,
                 head_units=None, input_bn=True):
        """Initialise the CNN-based DeepTCM model."""
        super().__init__()
        self.is_2d = is_2d
        self.input_bn = None
        if input_bn and not is_2d:
            self.input_bn = nn.BatchNorm1d(in_channels)

        self.denoise = FeatureDenoising(in_channels, filters[0], dropout, is_2d=is_2d)

        blocks = []
        ch = filters[0]
        Pool = (nn.AvgPool2d if is_2d else nn.AvgPool1d) if pooling_type == "avg" \
            else (nn.MaxPool2d if is_2d else nn.MaxPool1d)
        for i in range(1, len(filters)):
            blocks.append(ResNetBlock(ch, filters[i], kernel_sizes[i],
                                      attention=attention, is_2d=is_2d))
            blocks.append(nn.Dropout(dropout))
            blocks.append(Pool(2))
            ch = filters[i]
        self.blocks = nn.Sequential(*blocks)

        if head_units is None:
            head_units = [128, 64]
        head = []
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x):
        """Extract per-window CNN features from raw signal windows."""
        if self.input_bn is not None:
            x = self.input_bn(x)
        x = self.denoise(x)
        x = self.blocks(x)
        x = x.mean(dim=-1) if not self.is_2d else x.mean(dim=(-2, -1))
        x = self.head(x)
        return x

    def forward(self, x):
        """Forward pass for the CNN model variant."""
        x = self.encode(x)
        return self.output(x)


class TorchLSTM(nn.Module):
    """Stacked LSTM matching Keras create_lstm().

    Input: (B, T, C) for 1D or (B, T, F, C) for scalogram (F×C flattened).
    Layout: [InputBN →] LSTM × N [→ LayerNorm] → GAP → DenseHead.
    """

    def __init__(self, in_features, hidden_units, dropout=0.2,
                 bidirectional=True, head_units=None, input_bn=True,
                 layer_norm=True):
        """Initialise TorchLSTM.

        Args:
            in_features (int): Number of input features per timestep.
            hidden_units (list[int]): Hidden size for each stacked LSTM layer.
            dropout (float): Dropout probability in the dense head.
            bidirectional (bool): Wrap each LSTM in a bidirectional wrapper.
            head_units (list[int] | None): Dense head widths. Defaults to ``[128, 64]``.
            input_bn (bool): Apply BatchNorm1d to the input.
            layer_norm (bool): Apply LayerNorm after each LSTM layer output.
        """
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features) if input_bn else None
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm

        self.lstms = nn.ModuleList()
        self.lns = nn.ModuleList()
        inp = in_features
        for i, units in enumerate(hidden_units):
            self.lstms.append(nn.LSTM(inp, units, batch_first=True,
                                      bidirectional=bidirectional))
            d = units * (2 if bidirectional else 1)
            if layer_norm:
                self.lns.append(nn.LayerNorm(d))
            inp = d

        self.feat_dim = inp
        if head_units is None:
            head_units = [128, 64]
        head = []
        ch = inp
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x):
        """Extract per-window ResNet features."""
        if self.input_bn is not None:
            x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        for i, lstm in enumerate(self.lstms):
            is_last = (i == len(self.lstms) - 1)
            x, _ = lstm(x)
            if self.layer_norm and i < len(self.lns):
                x = self.lns[i](x)
            if is_last:
                x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward(self, x):
        """Forward pass for the ResNet model variant."""
        x = self.encode(x)
        return self.output(x)


class TorchBiGRU(nn.Module):
    """Stacked GRU matching Keras create_bigru().

    Same structure as TorchLSTM but with GRU cells.
    """

    def __init__(self, in_features, hidden_units, dropout=0.2,
                 bidirectional=True, head_units=None, input_bn=True,
                 layer_norm=True):
        """Initialise TorchBiGRU.

        Args:
            in_features (int): Number of input features per timestep.
            hidden_units (list[int]): Hidden size for each stacked GRU layer.
            dropout (float): Dropout probability in the dense head.
            bidirectional (bool): Wrap each GRU in a bidirectional wrapper.
            head_units (list[int] | None): Dense head widths. Defaults to ``[128, 64]``.
            input_bn (bool): Apply BatchNorm1d to the input.
            layer_norm (bool): Apply LayerNorm after each GRU layer output.
        """
        super().__init__()
        self.input_bn = nn.BatchNorm1d(in_features) if input_bn else None
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm

        self.grus = nn.ModuleList()
        self.lns = nn.ModuleList()
        inp = in_features
        for i, units in enumerate(hidden_units):
            self.grus.append(nn.GRU(inp, units, batch_first=True,
                                    bidirectional=bidirectional))
            d = units * (2 if bidirectional else 1)
            if layer_norm:
                self.lns.append(nn.LayerNorm(d))
            inp = d

        self.feat_dim = inp
        if head_units is None:
            head_units = [128, 64]
        head = []
        ch = inp
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x):
        """Extract temporal BiGRU features from the signal sequence."""
        if self.input_bn is not None:
            x = self.input_bn(x.transpose(1, 2)).transpose(1, 2)
        for i, gru in enumerate(self.grus):
            is_last = (i == len(self.grus) - 1)
            x, _ = gru(x)
            if self.layer_norm and i < len(self.lns):
                x = self.lns[i](x)
            if is_last:
                x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward(self, x):
        """Forward pass for the BiGRU model variant."""
        x = self.encode(x)
        return self.output(x)


class TorchCNNRNN(nn.Module):
    """CNN→RNN hybrid matching Keras create_cnn_lstm() / CNN_LSTM.

    Layout: CNN blocks → [CBAM] → [Flatten freq if 2D] → RNN → DenseHead.
    """

    def __init__(self, in_channels, filters, kernel_sizes, rnn_units,
                 dropout=0.2, pooling_type="avg", rnn_type="lstm",
                 attention=False, bidirectional=True, is_2d=False,
                 head_units=None, input_bn=True, layer_norm=True):
        """Initialise TorchCNNRNN.

        Args:
            in_channels (int): Number of input channels.
            filters (list[int]): Filter counts for each CNN block.
            kernel_sizes (list[int]): Kernel sizes for each CNN block.
            rnn_units (list[int]): Hidden sizes for each stacked RNN layer.
            dropout (float): Dropout probability used in both CNN and RNN heads.
            pooling_type (str): ``'avg'`` or ``'max'`` pooling after each CNN block.
            rnn_type (str): ``'lstm'`` or ``'gru'``.
            attention (bool): Insert a CBAMBlock after the CNN stack.
            bidirectional (bool): Use bidirectional RNN cells.
            is_2d (bool): Use 2-D CNN operations (for scalogram inputs).
            head_units (list[int] | None): Dense head widths. Defaults to ``[64, 32]``.
            input_bn (bool): Apply BatchNorm to the CNN input.
            layer_norm (bool): Apply LayerNorm after each RNN layer.
        """
        super().__init__()
        self.is_2d = is_2d
        self.input_bn = None
        if input_bn:
            BN = nn.BatchNorm2d if is_2d else nn.BatchNorm1d
            self.input_bn = BN(in_channels)

        cnn_layers = []
        ch = in_channels
        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            kk = (k, k) if is_2d else k
            cnn_layers.append(ConvBnRelu(ch, f, kk, padding="same" if not is_2d else "same",
                                         is_2d=is_2d))
            Pool = (nn.AvgPool2d if is_2d else nn.AvgPool1d) if pooling_type == "avg" \
                else (nn.MaxPool2d if is_2d else nn.MaxPool1d)
            ps = (2, 2) if is_2d else 2
            cnn_layers.append(Pool(ps))
            if dropout > 0:
                cnn_layers.append(nn.Dropout(dropout))
            ch = f
        self.cnn = nn.Sequential(*cnn_layers)

        self.cbam = CBAMBlock(ch, is_2d=is_2d) if attention else None

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.layer_norm = layer_norm
        self.rnns = nn.ModuleList()
        self.rnn_lns = nn.ModuleList()
        
        self._rnn_built = False
        self._rnn_units = rnn_units
        self._dropout = dropout
        self._head_units = head_units if head_units is not None else [64, 32]
        self._cnn_out_ch = ch

        ch = self._head_units[-1] if self._head_units else rnn_units[-1] * (2 if bidirectional else 1)
        self.out_dim = ch

        self.head = None
        self.output_layer = None

    def _build_rnn(self, rnn_input_size):
        """Lazily build RNN + head once input size is known."""
        RNN = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        inp = rnn_input_size
        for units in self._rnn_units:
            self.rnns.append(RNN(inp, units, batch_first=True,
                                 bidirectional=self.bidirectional))
            d = units * (2 if self.bidirectional else 1)
            if self.layer_norm:
                self.rnn_lns.append(nn.LayerNorm(d))
            inp = d

        ch = inp
        head = []
        for hu in self._head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(self._dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output_layer = nn.Linear(ch, 1)
        self._rnn_built = True

        device = next(self.cnn.parameters()).device
        for mod in [self.rnns, self.rnn_lns, self.head, self.output_layer]:
            if mod is not None:
                mod.to(device)

    def encode(self, x):
        """Extract signal features using the CNN and BiGRU encoder stack."""
        if self.input_bn is not None:
            x = self.input_bn(x)
        x = self.cnn(x)

        if self.cbam is not None:
            x = self.cbam(x)

        if self.is_2d:
            B, C, T, F = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, T, F * C)
        else:
            x = x.transpose(1, 2)

        if not self._rnn_built:
            self._build_rnn(x.shape[-1])

        for i, rnn in enumerate(self.rnns):
            is_last = (i == len(self.rnns) - 1)
            x, _ = rnn(x)
            if self.layer_norm and i < len(self.rnn_lns):
                x = self.rnn_lns[i](x)
            if is_last:
                x = x.mean(dim=1)
        x = self.head(x)
        return x

    def forward(self, x):
        """Forward pass for the champion model."""
        x = self.encode(x)
        return self.output_layer(x)


class ProcWrapper(nn.Module):
    """Wraps any backbone with concatenative process-param conditioning.

    Extracts encoder features from backbone.encode(signal), encodes proc params
    via ProcEncoder, concatenates, and feeds through a new regression head.
    """

    def __init__(self, backbone, proc_dim, head_units=None, dropout=0.2):
        """Initialise ProcWrapper.

        Args:
            backbone: Any encoder module exposing ``encode(x)`` and ``out_dim``.
            proc_dim (int): Dimensionality of the process-parameter vector.
            head_units (list[int] | None): Dense head widths after fusion.
                Defaults to ``[64, 32]``.
            dropout (float): Dropout probability in the fusion head.
        """
        super().__init__()
        self.backbone = backbone
        self.proc_enc = ProcEncoder(proc_dim)

        feat_dim = backbone.out_dim + self.proc_enc.out_dim
        if head_units is None:
            head_units = [64, 32]
        head = []
        ch = feat_dim
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x_signal, x_proc):
        """Encode signal and process parameters and return the fused feature vector.

        Args:
            x_signal: Signal input tensor compatible with the backbone encoder.
            x_proc (torch.Tensor): Process-parameter tensor of shape (B, P).

        Returns:
            torch.Tensor: Concatenated feature vector of shape (B, backbone.out_dim + proc_enc.out_dim).
        """
        feat = self.backbone.encode(x_signal)
        proc_feat = self.proc_enc(x_proc)
        return torch.cat([feat, proc_feat], dim=-1)

    def forward(self, x):
        """Forward pass for the LSTM model variant."""
        x_signal = x['x']
        x_proc = x['proc_data']
        feat = self.encode(x_signal, x_proc)
        feat = self.head(feat)
        return self.output(feat)


class FiLMWrapper(nn.Module):
    """Wraps any backbone with FiLM conditioning from process params.

    Applies (γ+1)⊙features + β before the regression head.
    """

    def __init__(self, backbone, proc_dim, head_units=None, dropout=0.2):
        """Initialise FiLMWrapper.

        Args:
            backbone: Any encoder exposing ``encode(x)`` and ``out_dim``.
            proc_dim (int): Dimensionality of the process-parameter vector.
            head_units (list[int] | None): Dense head widths. Defaults to ``[64]``.
            dropout (float): Dropout probability in the fusion head.
        """
        super().__init__()
        self.backbone = backbone
        self.film = FiLMLayer(backbone.out_dim, proc_dim)

        if head_units is None:
            head_units = [64]
        ch = backbone.out_dim
        head = []
        for hu in head_units:
            head.extend([nn.Linear(ch, hu), nn.ReLU(), nn.Dropout(dropout)])
            ch = hu
        self.head = nn.Sequential(*head)
        self.out_dim = ch
        self.output = nn.Linear(ch, 1)

    def encode(self, x_signal, x_proc):
        """Apply FiLM conditioning and return the modulated feature vector.

        Args:
            x_signal: Signal input tensor compatible with the backbone encoder.
            x_proc (torch.Tensor): Process-parameter tensor of shape (B, P).

        Returns:
            torch.Tensor: FiLM-modulated feature vector of shape (B, backbone.out_dim).
        """
        feat = self.backbone.encode(x_signal)
        feat = self.film(feat, x_proc)
        return feat

    def forward(self, x):
        """Forward pass for the stacked-CNN model variant."""
        x_signal = x['x']
        x_proc = x['proc_data']
        feat = self.encode(x_signal, x_proc)
        feat = self.head(feat)
        return self.output(feat)


class PlainWrapper(nn.Module):
    """Wraps a backbone that doesn't use proc_data — just passes signal through.

    Accepts the dict input convention and discards proc_data.
    """

    def __init__(self, backbone):
        """Initialise PlainWrapper.

        Args:
            backbone: Any encoder module with a callable ``forward(x)``.
        """
        super().__init__()
        self.backbone = backbone
        self.out_dim = backbone.out_dim

    def forward(self, x):
        """Aggregate window-level predictions to run-level via mean pooling."""
        x_signal = x['x']
        return self.backbone(x_signal)


def _to_channels_first_1d(x):
    """(B, T, C) → (B, C, T) for Conv1d."""
    return x.transpose(1, 2)


def _to_channels_first_2d(x):
    """(B, T, F, C) → (B, C, T, F) for Conv2d."""
    return x.permute(0, 3, 1, 2)


def get_torch_model(model_name, input_shape, hps, proc_shape=None):
    """
    Factory function to create PyTorch models for DeepTCM CL experiments.

    Mirrors the Keras get_model() dispatcher. Parses _Proc / _Film suffixes,
    dispatches to the correct backbone, and wraps with conditioning if needed.

    Args:
        model_name (str): Architecture name, e.g. "ResNet", "LSTM", "ResNet_Proc",
            "RobustResNet_Film", "RobustCNN_LSTM_Proc".
        input_shape (tuple): Signal shape WITHOUT batch dim.
            1D: (T, C) — e.g. (500, 6)
            2D scalogram: (T, F, C) — e.g. (9, 33, 6)
        hps (dict): Hyperparameters (same keys as Keras get_model).
        proc_shape (tuple | None): Process parameter shape, e.g. (4,).
            Required for _Proc and _Film variants.

    Returns:
        nn.Module accepting forward(x) where x={'x': tensor, 'proc_data': tensor}.
    """
    original_name = model_name

    use_proc = False
    use_film = False

    if "_Film" in model_name and "Ensemble" not in model_name:
        use_film = True
        model_name = model_name.replace("_Film_Stats", "").replace("_Film", "")
    elif model_name.endswith("_Proc") and "Ensemble" not in model_name:
        use_proc = True
        model_name = model_name.replace("_Proc", "")

    if (use_proc or use_film) and proc_shape is None:
        raise ValueError(
            f"proc_shape required for '{original_name}' (has _Proc or _Film suffix)"
        )

    is_2d = len(input_shape) == 3
    in_channels = input_shape[-1]

    dropout = hps.get("dropout", 0.2)

    backbone = None
    if model_name == "CNN":
        base = hps.get("filters_base") or 8
        n_cnn = hps.get("cnn_layers") or 2
        kernel_tmpl = [5, 5, 3, 3, 3]
        kernels = (kernel_tmpl + [3] * max(0, n_cnn - len(kernel_tmpl)))[:n_cnn]
        filters = [base * (2 ** i) for i in range(n_cnn)]

        backbone = _ChannelsFirstCNN(
            TorchCNN(in_channels, filters, kernels, dropout=dropout,
                     pooling_type=hps.get("pooling", "avg"), is_2d=is_2d),
            is_2d=is_2d,
        )

    elif model_name == "LSTM":
        n_layers = hps.get("lstm_layers", 2)
        units = [hps.get("lstm_units", 64)] * n_layers
        in_feat = in_channels
        if is_2d:
            in_feat = input_shape[1] * input_shape[2]  # F * C

        backbone = _ScalogramFlattenLSTM(
            TorchLSTM(in_feat, units, dropout=dropout,
                      bidirectional=hps.get("bidirectional", True)),
            is_2d=is_2d,
        )

    elif model_name == "BiGRU":
        n_layers = hps.get("lstm_layers", 2)
        units = [hps.get("lstm_units", 64)] * n_layers
        in_feat = in_channels
        if is_2d:
            in_feat = input_shape[1] * input_shape[2]

        backbone = _ScalogramFlattenGRU(
            TorchBiGRU(in_feat, units, dropout=dropout,
                       bidirectional=hps.get("bidirectional", True)),
            is_2d=is_2d,
        )

    elif "CNN_LSTM" in model_name:
        base = hps.get("filters_base") or 8
        n_cnn = hps.get("cnn_layers") or 2
        kernel_tmpl = [5, 5, 3, 3, 3]
        kernels = (kernel_tmpl + [3] * max(0, n_cnn - len(kernel_tmpl)))[:n_cnn]
        filters = [base * (2 ** i) for i in range(n_cnn)]
        n_rnn = hps.get("lstm_layers", 1)
        rnn_units = [hps.get("lstm_units", 64)] * n_rnn
        attention = "Robust" in model_name

        backbone = _ChannelsFirstCNNRNN(
            TorchCNNRNN(in_channels, filters, kernels, rnn_units,
                        dropout=dropout,
                        pooling_type=hps.get("pooling", "avg"),
                        rnn_type=hps.get("rnn_type", "lstm"),
                        attention=attention,
                        bidirectional=hps.get("bidirectional", True),
                        is_2d=is_2d),
            is_2d=is_2d,
        )

    elif "ResNet" in model_name:
        base = hps.get("filters_base") or 8
        n_cnn = hps.get("cnn_layers") or 2
        kernel_tmpl = [5, 5, 3, 3, 3]
        kernels = (kernel_tmpl + [3] * max(0, n_cnn - len(kernel_tmpl)))[:n_cnn]
        filters = [base * (2 ** i) for i in range(n_cnn)]
        attention = "Robust" in model_name

        backbone = _ChannelsFirstResNet(
            TorchResNet(in_channels, filters, kernels, dropout=dropout,
                        pooling_type=hps.get("pooling", "avg"),
                        attention=attention, is_2d=is_2d),
            is_2d=is_2d,
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    proc_dim = proc_shape[-1] if proc_shape is not None else 0

    if use_film:
        model = FiLMWrapper(backbone, proc_dim, dropout=dropout)
    elif use_proc:
        model = ProcWrapper(backbone, proc_dim, dropout=dropout)
    else:
        model = PlainWrapper(backbone)

    return model


class _ChannelsFirstCNN(nn.Module):
    """Wraps TorchCNN: converts (B,T,C)→(B,C,T) or (B,T,F,C)→(B,C,T,F)."""

    def __init__(self, cnn, is_2d=False):
        """Initialise _ChannelsFirstCNN.

        Args:
            cnn (TorchCNN): Pre-built CNN module (expects channels-first input).
            is_2d (bool): Apply 2-D channels-first conversion when True.
        """
        super().__init__()
        self.cnn = cnn
        self.is_2d = is_2d
        self.out_dim = cnn.out_dim

    def encode(self, x):
        """Encode one signal modality."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.cnn.encode(x)

    def forward(self, x):
        """Forward pass through the signal encoder."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.cnn(x)


class _ChannelsFirstResNet(nn.Module):
    """Wraps TorchResNet with channels-first conversion."""

    def __init__(self, resnet, is_2d=False):
        """Initialise _ChannelsFirstResNet.

        Args:
            resnet (TorchResNet): Pre-built ResNet module.
            is_2d (bool): Apply 2-D channels-first conversion when True.
        """
        super().__init__()
        self.resnet = resnet
        self.is_2d = is_2d
        self.out_dim = resnet.out_dim

    def encode(self, x):
        """Encode process parameters."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.resnet.encode(x)

    def forward(self, x):
        """Forward pass through the proc-param encoder."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.resnet(x)


class _ChannelsFirstCNNRNN(nn.Module):
    """Wraps TorchCNNRNN with channels-first conversion."""

    def __init__(self, cnnrnn, is_2d=False):
        """Initialise _ChannelsFirstCNNRNN.

        Args:
            cnnrnn (TorchCNNRNN): Pre-built CNN-RNN module.
            is_2d (bool): Apply 2-D channels-first conversion when True.
        """
        super().__init__()
        self.cnnrnn = cnnrnn
        self.is_2d = is_2d
        self.out_dim = cnnrnn.out_dim

    def encode(self, x):
        """Encode and fuse signal and proc-param modalities."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.cnnrnn.encode(x)

    def forward(self, x):
        """Forward pass through the multimodal fusion encoder."""
        x = _to_channels_first_2d(x) if self.is_2d else _to_channels_first_1d(x)
        return self.cnnrnn(x)


class _ScalogramFlattenLSTM(nn.Module):
    """Wraps TorchLSTM: flattens scalogram (B,T,F,C)→(B,T,F*C) if 2D."""

    def __init__(self, lstm, is_2d=False):
        """Initialise _ScalogramFlattenLSTM.

        Args:
            lstm (TorchLSTM): Pre-built LSTM module expecting a 3-D input.
            is_2d (bool): Flatten the F×C frequency-channel dimensions when True.
        """
        super().__init__()
        self.lstm = lstm
        self.is_2d = is_2d
        self.out_dim = lstm.out_dim

    def _prep(self, x):
        """Prepare input tensor for GRU: handle 3-D and 4-D inputs."""
        if self.is_2d:
            B, T, F, C = x.shape
            x = x.reshape(B, T, F * C)
        return x

    def encode(self, x):
        """Encode the input sequence using the bidirectional GRU."""
        return self.lstm.encode(self._prep(x))

    def forward(self, x):
        """Forward pass through the BiGRU encoder."""
        return self.lstm(self._prep(x))


class _ScalogramFlattenGRU(nn.Module):
    """Wraps TorchBiGRU: flattens scalogram (B,T,F,C)→(B,T,F*C) if 2D."""

    def __init__(self, gru, is_2d=False):
        """Initialise _ScalogramFlattenGRU.

        Args:
            gru (TorchBiGRU): Pre-built GRU module expecting a 3-D input.
            is_2d (bool): Flatten the F×C frequency-channel dimensions when True.
        """
        super().__init__()
        self.gru = gru
        self.is_2d = is_2d
        self.out_dim = gru.out_dim

    def _prep(self, x):
        """Prepare input tensor for LSTM: handle 3-D and 4-D inputs."""
        if self.is_2d:
            B, T, F, C = x.shape
            x = x.reshape(B, T, F * C)
        return x

    def encode(self, x):
        """Encode the input sequence using the LSTM."""
        return self.gru.encode(self._prep(x))

    def forward(self, x):
        """Forward pass through the LSTM encoder."""
        return self.gru(self._prep(x))
