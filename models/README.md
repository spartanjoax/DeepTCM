# models/

Model definitions and the AutoML search infrastructure. Contains two parallel implementations: a **Keras** model zoo used by `train.py` and `train_ak.py`, and a **PyTorch** port used by the Avalanche-based CL scripts.

## Architecture families

| Name | Type | Notes |
|---|---|---|
| `LSTM` | LSTM | Temporal encoder |
| `BiGRU` | Bidirectional GRU | Temporal encoder |
| `CNN` | CNN | Spatial encoder |
| `ResNet` | ResNet | CNN spatial encoder with skips |
| `ResNet_Proc` | ResNet + FiLM/Proc | Process-param conditioning |
| `RobustResNet` | ResNet + CBAM | Attention mechanism |
| `RobustResNet_Proc` | RobustResNet + FiLM/Proc | Process-param conditioning + attention mechanism |
| `CNN_LSTM` | CNN + LSTM | Tempo-spatial encoder |
| `CNN_LSTM_Proc` | CNN_LSTM_ + FiLM/Proc | Process-param conditioning |
| `RobustCNN_LSTM` | CNN_LSTM + CBAM | Attention mechanism |
| `RobustCNN_LSTM_Proc` | RobustCNN_LSTM + FiLM/Proc | Process-param conditioning + attention mechanism |

### FiLM conditioning

Feature-wise Linear Modulation (FiLM) injects process parameters (depth-of-cut, feed, material, cutting speed) as affine scale+shift transforms on intermediate feature maps. It is effective when combined with an STFT front-end because the scalogram CNN extracts frequency-bin features without encoding temporal metadata. The process parameters provide information the scalogram alone cannot recover.

FiLM is **not** beneficial when applied to raw time-domain signals with a CNN+RNN encoder — the recurrent encoder already captures this information implicitly.

## Usage examples

```python
import json
from models import get_model

with open("configs/champion_timefreq_domain.json") as f:
    hps = json.load(f)

# Keras model (6 NASA channels, 3 process params)
model = get_model(hps, n_channels=6, n_proc=3)

# PyTorch model for CL (e.g. AC group = 1 channel)
from models.dl_models_torch import get_torch_model
torch_model = get_torch_model("RobustCNN_LSTM_Proc", hps, n_channels=1, n_proc=3)
```
