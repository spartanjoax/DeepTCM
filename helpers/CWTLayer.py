"""
Backend-agnostic CWT layer for Keras 3.x
Works with TensorFlow, PyTorch and JAX backends.
"""
import numpy as np
from keras import Layer, ops
from keras.utils import register_keras_serializable
#import matplotlib.pyplot as plt
import pywt

@register_keras_serializable(package="cwt")
class CWTLayer(Layer):
    """
    Continuous wavelet transform expressed as 1-D convolutions.
    Input  : (batch, time, channels)
    Output : (batch, height, width, channels)   # height = nb_scales, width = time
    """

    def __init__(self,
                 wavelet: str = "cmor",
                 scales: int = 128,
                 kernel_length_factor: int = 10,   # kernel width = factor * largest_scale
                 output_size: tuple = (224, 224),
                 **kwargs):
        super().__init__(**kwargs)
        self.wavelet = wavelet
        self.scales = scales
        self.kernel_length_factor = kernel_length_factor
        self.out_h, self.out_w = output_size

# ------------------------------------------------------------------
# 1.  build()  --  add one integrated-wavelet buffer
# ------------------------------------------------------------------
    def build(self, input_shape):
        self.t_max, self.ch = input_shape[1], input_shape[2]

        # --- same scale vector you already use ---
        max_scale = self.t_max // (2 * self.kernel_length_factor)
        self.scales_vec = np.logspace(
            np.log10(2), np.log10(max_scale), self.scales
        ).astype(np.float32)

        # ----------  integrated wavelet  ----------
        precision = 12

        # build integrated wavelet (complex)
        int_psi_base, x_master, wavelet = self._integrated_wavelet(self.wavelet, precision)
        int_psi_base = ops.conj(int_psi_base) if wavelet.complex_cwt else int_psi_base
        
        # we store these two vectors as non-trainable weights
        self.x_master = self.add_weight(
            name="x_master", 
            shape=x_master.shape, 
            initializer="zeros", 
            trainable=False,
        )
        self.int_psi_base = self.add_weight(
            name="int_psi_base",
            shape=int_psi_base.shape,
            initializer="zeros",
            trainable=False,
            #dtype=int_psi_base.dtype,
        )
        self.x_master.assign(x_master)        
        self.int_psi_base.assign(int_psi_base)

        super().build(input_shape)

# ------------------------------------------------------------------
# 2.  integrated wavelet helper  (morlet example)
# ------------------------------------------------------------------    
    def _integrated_wavelet(self, wavelet_name="morse", precision=10):
        """
        Use PyWavelets to compute the integrated base wavelet.
        """
        # 1. Generate wavelet function from PyWT
        wavelet = pywt.ContinuousWavelet(wavelet_name)
        psi, x = wavelet.wavefun(precision)
    
        # 2. Integrate once over time
        dt = x[1] - x[0]
        int_psi = np.cumsum(psi) * dt
        
        # 3. Center and normalize
        int_psi -= np.mean(int_psi)
        
        int_psi /= np.sqrt(np.sum(np.abs(int_psi) ** 2))
        
        #plt.plot(np.real(int_psi), label="real")
        #plt.plot(np.imag(int_psi), label="imag")
        #plt.savefig(f"CWT_images/integrated_wavelet_2.png")
        #plt.show()

        return int_psi.astype(np.complex64), x.astype(np.float32), wavelet
        
# ------------------------------------------------------------------
# 3.  call()  --  replicate PyWT step-by-step
# ------------------------------------------------------------------
    def call(self, inputs):
        x = ops.cast(inputs, self.compute_dtype)          # (B, T, C)
        B = ops.shape(x)[0]
        T = self.t_max
        C = self.ch

        # ---------- 0.  pad exactly as PyWT does  ----------
        width = ops.shape(self.x_master)[0]
        pad = (width - 1) // 2
        pad = ops.minimum(pad, T - 1)
        x = ops.pad(x, [[0, 0], [pad, pad], [0, 0]], mode="reflect")

        # ---------- 1.  channel-first for vectorised loop ----------
        x = ops.transpose(x, [2, 0, 1])                  # (C, B, T)
        x = ops.reshape(x, [C * B, T + 2 * pad])         # (C·B, T_pad)

        # ---------- 2.  allocate output  ----------
        out = ops.zeros((C * B, len(self.scales_vec), T), dtype=self.compute_dtype)

        # ---------- 3.  scale loop  ----------
        coef_list = []                       # ← list, same as PyWT
        for i, scale in enumerate(self.scales_vec):
            # interpolate integrated wavelet  (same as before)
            step = self.x_master[1] - self.x_master[0]
            j = ops.arange(scale * (self.x_master[-1] - self.x_master[0]) + 1, dtype="float32") / (scale * step)
            j = ops.cast(ops.floor(j), "int32")
            j = ops.clip(j, 0, ops.shape(self.int_psi_base)[0] - 1)
            int_psi_scale = ops.flip(ops.take(self.int_psi_base, j, axis=0), axis=0)

            # ---- 4.  convolve  ----
            kernel = ops.reshape(int_psi_scale, [-1, 1, 1])   # (W, 1, 1)
            
            # prepare input for conv: (batch, length, channels)
            x = ops.reshape(x, [C * B, T + 2 * pad, 1])  # (batch, length, channels)
            
            conv = ops.conv(
                x,kernel, strides=1, padding="VALID",data_format="channels_last",
            )
            conv = ops.squeeze(conv, axis=-1)  # remove out_ch dimension -> (C*B, new_length)

            # ---- 5.  differentiate  ----
            # np.diff along last axis -> conv[:, 1:] - conv[:, :-1]
            conv_right = ops.take(conv, ops.arange(1, ops.shape(conv)[-1]), axis=-1)
            conv_left  = ops.take(conv, ops.arange(0, ops.shape(conv)[-1] - 1), axis=-1)
            coef = conv_right - conv_left   # shape (C*B, conv_len-1)

            # ---- 6.  normalise  ----
            coef = -ops.sqrt(ops.cast(scale, coef.dtype)) * coef

            # ---- 7.  trim to original length  ----
            d = (ops.shape(coef)[-1] - T) // 2
            # now select indices [d, d+T) in a tensor-safe way
            start = ops.cast(ops.maximum(d, 0), "int32")
            end = ops.minimum(start + T, ops.shape(coef)[-1])
            idx = ops.arange(start, end, dtype="int32")  # shape (T,)
            coef = ops.take(coef, idx, axis=-1)                # -> (C*B, T)

            if ops.shape(coef)[-1] != self.t_max:
                raise ValueError("The length of the coef is not the same as the time:", coef)
                
            coef_list.append(coef)                    # ← list append

        # ---------- 8.  stack once  ----------
        out = ops.stack(coef_list, axis=1)            # (C·B, scales, T)
        # reshape while keeping 1st axis unknown
        out = ops.reshape(out, [-1, self.scales, T])  # (C*B, scales, T)
        # split channel & batch  (same trick you used before)
        out = ops.reshape(out, [C, -1, self.scales, T])  # (C, B, scales, T)
        out = ops.transpose(out, [1, 2, 3, 0])        # (B, scales, T, C)

        # ---------- 9.  optional resize  ----------
        if (self.out_h, self.out_w) != (len(self.scales_vec), T):
            out = ops.image.resize(out, [self.out_h, self.out_w])

        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "wavelet": self.wavelet,
            "scales": self.scales,
            "kernel_length_factor": self.kernel_length_factor,
            "output_size": (self.out_h, self.out_w),
        })
        return cfg

__all__ = [
    'CWTLayer',
]