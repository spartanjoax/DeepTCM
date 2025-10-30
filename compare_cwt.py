"""
compare_cwt_pywt_vs_keras3.py
Single-channel test:  PyWT  vs  your Keras-3 CWTLayer
"""
import numpy as np
import pywt
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from helpers import CWTLayer   # <- your file with the layer

# ------------------------------------------------------------------
# 1.  Test signal
# ------------------------------------------------------------------
fs = 250.0
T = 2.0                         # 2 s  ->  500 samples
t = np.arange(0, T, 1/fs)
# 10 Hz + 50 Hz + burst
x = (0.5*np.sin(2*np.pi*10*t) +
     0.3*np.sin(2*np.pi*50*t) +
     0.2*np.exp(-((t-1)/0.1)**2)*np.sin(2*np.pi*80*t))
x = x.astype(np.float32)

# ------------------------------------------------------------------
# 2.  Common parameters
# ------------------------------------------------------------------
n_scales = 224
scales = np.arange(1, n_scales+1)                    # 1 … 64  (same for both)
wavelet_name = "morl"                      # PyWT name
# beta/gamma only used by your layer if you pick "morse"
output_img = (224, 500)                       # keep original time length

# ------------------------------------------------------------------
# 3.  PyWavelets reference
# ------------------------------------------------------------------
coef_pywt, _ = pywt.cwt(x, scales, wavelet_name, sampling_period=1/fs)
# coef shape:  (64, 500)  – magnitude already |complex|
mag_pywt = np.abs(coef_pywt)

# ------------------------------------------------------------------
# 4.  Keras-3 layer
# ------------------------------------------------------------------
layer = CWTLayer(wavelet=wavelet_name,
                 scales=len(scales),
                 output_size=output_img,
                 kernel_length_factor=10)
# build once
_ = layer(tf.zeros((1, len(x), 1)))
# run
x_in = x[None, :, None]          # (1, 500, 1)
mag_keras = layer(x_in).numpy().squeeze()   # (64, 500)

# ------------------------------------------------------------------
# 5.  Difference
# ------------------------------------------------------------------
diff = np.abs(mag_pywt - mag_keras).mean()
print(f"Mean absolute difference  (PyWT vs Keras):  {diff:.4e}")

# ------------------------------------------------------------------
# 6.  Quick visual check
# ------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].imshow(mag_pywt, aspect='auto', cmap='jet')
ax[0].set_title("PyWavelets")
ax[1].imshow(mag_keras, aspect='auto', cmap='jet', origin='upper')
ax[1].set_title("Keras-3 layer")
ax[2].imshow(np.abs(mag_pywt - mag_keras), aspect='auto', cmap='gray')
ax[2].set_title("|difference|")
plt.tight_layout()
plt.savefig(f"CWT_images/wavelets.png")
plt.show()

row = 25
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.plot(mag_pywt[row], label='PyWT')
ax.plot(mag_keras[row], '--', label='Keras')
plt.legend()
plt.savefig(f"CWT_images/wavelets_line.png")