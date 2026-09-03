# DeepTCM — Deep Continual Learning for Tool Condition Monitoring

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5-orange)
![Keras 3](https://img.shields.io/badge/Keras-3.x-red)
![Avalanche 0.6](https://img.shields.io/badge/Avalanche-0.6-green)

This repository covers three complementary contributions: (1) expert-defined deep learning models, (2) joint AutoML hyperparameter search, and (3) a continual learning evaluation for cross-machine and cross-material tool wear adaptation.

---

## Why it matters

Tool condition monitoring (TCM) is a prerequisite for autonomous machining: a spindle running with a worn insert produces scrapped parts and risks machine damage. Direct measurement of flank wear (VB) requires stopping the machine. Thus, real-time prediction from signals recorded during cutting is used to reduce downtime.

The hard problem is **generalisation**. A model trained on one machine, material, or cutting condition degrades on another. Collecting enough labelled data for each new deployment is economically prohibitive. This repository aims to address both challenges by:

1. Establishing a reliable DL performance ceiling on the [NASA Ames/UC Berkeley face-milling dataset](https://data.nasa.gov/dataset/milling-wear) under a strict cross-condition protocol.
2. Demonstrating that a model pre-trained on NASA data can be adapted to another dataset (MU-TCM) via continual learning.
3. Using only **internal CNC signals** (no external sensors needed) via continual learning.

---

## Datasets

### NASA Ames Milling Dataset

A widely used benchmark for face-milling TCM.

- **16 milling cases** (case 6 excluded — documented anomaly)
- 6 raw signals at 250 Hz: AC and DC spindle motor currents, as well as vibrations and acoustic emission from the spindle and table.
- Target: flank wear VB (mm), capped at 0.45 mm.
- **Fixed paper split** (Zheng 2017, https://doi.org/10.1109/ICPHM.2017.7998311): train cases {1–5, 7–10, 13–14} and test cases {11, 12, 15, 16}. Case 6 excluded.

Download: [NASA Ames/UC Berkeley face-milling dataset](https://data.nasa.gov/dataset/milling-wear)

Place the raw `.mat` file at `data/mill.mat` and run `python data/interactive_mat_to_csv.py` to convert it. The binary cache (`data/nasa.bin`) is generated automatically on first training run. 

> If `python data/interactive_mat_to_csv.py` is not run manually, an optimistic, fixed segmentation will be performed to remove the entry and exit cut segments of the signals.

### MU-TCM (Mondragon Unibertsitatea CNC Milling Dataset)

A new dataset collected on a LAGUN GVC 1000 CNC at Mondragon Unibertsitatea for the continual learning experiments.

- 80 mm face mill, single SPKR M55 insert.
- 8 cutting conditions (2 materials × 2 speeds × 2 feeds): cast iron GG30 (dry) and stainless steel 316L (MQL).
- 16 internal CNC signals at 250 Hz (spindle/axis current, torque, power, and position). 6 external signals at 50 kHz (vibrations and cutting forces) and 2 at 1 MHz (accoustic emission) from the table.
- 67 total runs, VB measured at 4 levels (0.0, 0.1, 0.2, 0.3 mm).

Download: [MU-TCM dataset](https://hdl.handle.net/20.500.11984/6926) (Peralta et al., 2025). Place the directory at `data/MU-TCM/`.

---

## Repository structure

```
DeepTCM/
├── train.py                 # Expert model training
├── train_ak.py              # AutoML joint HP search
├── train_cl.py              # Continual learning training
├── train_separate.py        # Per-experience independent baseline
├── Dockerfile               # GPU-enabled reproducible environment
├── requirements.txt
├── configs/                 # Champion hyperparameter configs
├── continual/               # Avalanche CL strategies and regression metrics
├── data/                    # Data loaders for NASA and MU-TCM datasets
├── helpers/                 # Signal preprocessing and training utilities
├── models/                  # Model zoo (Keras + PyTorch), AutoML search space
```

---

## Installation

### Option A: conda + pip

```bash
conda create -n deeptcm python=3.11
conda activate deeptcm
pip install -r requirements.txt
```

Key dependencies (see `requirements.txt` for pinned versions):
- `torch >= 2.5.1` — PyTorch backend
- `keras >= 3.0` — with `KERAS_BACKEND=torch`
- `keras-tuner` — AutoML HP search
- `avalanche-lib == 0.6.0` (from [spartanjoax/avalanche_multimodal](https://github.com/spartanjoax/avalanche_multimodal) — adds multimodal dict input and regression support)
- `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`

### Option B: Docker

Build the image:
```bash
docker build -t deeptcm .
```

Run with GPU access and a mounted data directory:
```bash
docker run -it --gpus all -p 8889:8889 \
  -v /path/to/your/data:/workspace/DeepTCM/data \
  --runtime=nvidia deeptcm
```

The container starts a Jupyter Notebook server at `http://localhost:8889`.

---

## How to run

> **All scripts expect the `KERAS_BACKEND=torch` environment variable.** It is set automatically inside each script; no manual action is needed.

### 1. Expert model training (`train.py`)

Trains hand-designed architectures using LOCO-CV on the NASA dataset. The hyperparameter config (produced by `train_ak.py`) is loaded from a JSON file via `--hp_file`.

```bash
# LOCO-CV run on all signal groups with default champion config
python train.py

# Run a specific signal group with a custom config
python train.py --run AC --hp_file configs/champion_model.json

# Evaluate on signal group 'all' using a specific config file
python train.py --run all --hp_file configs/champion_model.json
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `-e` / `--epochs` | 3000 | Training epochs |
| `-sw` / `--window` | 250 | Sliding window size (samples) |
| `-ss` / `--stride` | 125 | Sliding window stride |
| `-hp` / `--hp_file` | `configs/champion_model.json` | Path to HP JSON config (output of `train_ak.py`) |
| `-run` / `--run` | `''` (→ `all`) | Signal group: `all`, `AC`, `DC`, `ACDC`, `AC_table`, `DC_table` |
| `-r` / `--retrain` | False | If set, retrain even if checkpoints exist |
| `-t` / `--test` | False | Test mode (reduced model set for quick validation) |
| `-sched` / `--scheduler` | `plateau` | LR scheduler: `plateau` or `cosine` |
| `-bs` / `--batch_size` | 16 | Batch size |
| `--experiment` | None | Output folder prefix label |

### 2. AutoML hyperparameter search (`train_ak.py`)

Runs the custom `PreprocessingRandom` joint search over preprocessing choices, architecture, and training hyperparameters on the NASA dataset. The best config is saved to `automl/<run>/champion_model.json`, which can then be copied to `configs/` and used with `train.py`.

```bash
# 50 random trials for 300 epochs
python train_ak.py -o 50 -e 300

# Search time-frequency representations using LOCO-CV scoring
python train_ak.py -o 50 --search_mode timefreq_only --use_loco_cv
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `-e` / `--epochs` | 300 | Training epochs per trial |
| `-o` / `--num_trials` | 50 | Number of trials |
| `-w` / `--window_size` | 10 | Moving-average denoising window size |
| `-m` / `--model_type` | `sota_search` | Model family to search (`sota_search`, `cnn`, `lstm`, `resnet`, ...) |
| `--tuner_type` | `random` | Tuner backend: `random` or `hyperband` |
| `--search_mode` | `split` | `split` (time then time-freq), `time_only`, `timefreq_only`, or `full` |
| `--use_loco_cv` | False | Use 11-fold LOCO-CV for champion verification instead of fixed val split |
| `-sw` / `--sliding_window_size` | 250 | Sliding window size (samples) |
| `-ss` / `--sliding_window_stride` | 125 | Sliding window stride |
| `--downsample` | 1 | Decimation factor applied before windowing |

**Hyperparameters searched:**

| HP | Type | Options |
|---|---|---|
| `scalogram` | Choice | `none`, `stft`, `cwt` (scope: `search_mode`) |
| `noise_reduction` | Choice | `moving_average`, `rms`, `none` |
| `jitter_sigma` | Float | 0.0 – 0.05 |
| `arch_type` | Choice | `rnn_only`, `cnn_rnn` |
| `rnn_type` | Choice | `lstm`, `bigru` |
| `lstm_units` | Choice | 64, 128, 256 |
| `lstm_layers` | Choice | 2, 3 |
| `filters_base` | Choice | 8, 16, 32 (only when `arch_type=cnn_rnn`) |
| `cnn_layers` | Choice | 2, 3 (only when `arch_type=cnn_rnn`) |
| `dropout` | Float | 0.0 – 0.5 |
| `weight_decay` | Choice | 1e-5, 1e-4, 1e-3 |
| `learning_rate` | Choice | 1e-4, 5e-4, 1e-3 |
| `conditioning` | Choice | `no`, `film`, `proc` |
| `pooling` | Fixed | `avg` |

### 3. Continual learning training (`train_cl.py`)

Trains the CL framework across six scenarios and six strategies using the [Avalanche](https://github.com/spartanjoax/avalanche_multimodal) library. Results (RMSE, R², forgetting) are written per seed to `continual/results/`.

```bash
# Run all scenarios with all strategies (auto-generated seed)
python train_cl.py

# Run specific scenarios and strategies
python train_cl.py --scenarios scenario_1 scenario_3 --strategies ewc si

# Run with a fixed seed and no resume (re-run everything)
python train_cl.py --seed 42 --no-resume
```

**Scenarios** (defined in `train_cl.py`):

| Scenario | Description |
|---|---|
| `scenario_1` | NASA → MU-TCM SS316L → MU-TCM GG30; external signals (AC, AC_table, DC, DC_table) |
| `scenario_2` | NASA → 8 individual MU-TCM conditions (SS then GG30); external signals |
| `scenario_3` | MU-TCM GG30 → 4 SS316L conditions; internal CNC signals only |
| `scenario_4` | MU-TCM SS316L → 4 GG30 conditions; internal CNC signals only |
| `scenario_5` | MU-TCM GG30 → MU-TCM SS316L → NASA; external signals |
| `scenario_6` | 8 MU-TCM conditions in reversed order; internal CNC signals only |

**Strategies**: `Naive` (fine-tuning), `Cumulative` (oracle upper bound), `EWC`, `SI`, `MAS`, and `AGEM`.

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--epochs` | 200 | Max training epochs per experience |
| `--patience` | 20 | Early-stopping patience |
| `--window-size` | 500 | Sliding window size (samples) |
| `--window-stride` | 500 | Sliding window stride (no overlap) |
| `--scenarios` | all except scenario_2 | One or more of `scenario_1` … `scenario_6` |
| `--strategies` | all | One or more of `cumulative`, `naive`, `ewc`, `si`, `mas`, `agem` |
| `--seed` | None (auto) | Random seed; run multiple times without this flag for multi-seed evaluation |
| `--resume` / `--no-resume` | `--resume` | Skip already-completed experiments |

### 4. Per-experience baseline (`train_separate.py`)

Trains a fresh independent model for each experience with no weight transfer. Serves as the no-CL oracle (upper bound per experience, lower bound for stability).

```bash
# Run on all signal groups and all datasets
python train_separate.py

# Run internal signals only on MU-TCM datasets
python train_separate.py --signal_groups internals --datasets muss muci
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--signal_groups` | `all` | Signal group(s) to run (e.g. `all`, `AC`, `internals`) |
| `--datasets` | `all` | Datasets to include: `nasa`, `muss` (MU-TCM SS316L), `muci` (MU-TCM GG30), or `all` |
| `--seed` | 42 | Random seed |
| `--epochs` | 1000 | Max training epochs per model |
| `--patience` | 20 | Early-stopping patience |
| `--window_size` | 500 | Sliding window size (samples) |
| `--window_stride` | 500 | Sliding window stride |
| `--resume` / `--no-resume` | `--resume` | Skip already-completed experiments |

---

## Acknowledgements

This work was developed at the [Software and Systems Engineering](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/software-systems-engineering) and [High-Performance Machining](https://www.mondragon.edu/en/research-transfer/engineering-technology/research-and-transfer-groups/high-performance-machining) groups at Mondragon University, as part of the [Digital Manufacturing and Design Training Network](https://dimanditn.eu/es/home).

This project has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 814078 (DIMAND) and by the Department of Education, Universities and Research of the Basque Government under the projects Ikerketa Taldeak (Grupo de Ingeniería de Software y Sistemas IT1519-22 and Grupo de investigación de Mecanizado de Alto Rendimiento IT1443-22).

---

## Contributors

<a href="https://github.com/spartanjoax"><img src="https://avatars.githubusercontent.com/u/29443664?v=4" title="José Joaquín Peralta Abadía" width="80" height="80"></a>

---

## Cite

If you use this code or the associated datasets, please cite:

```bibtex
@phdthesis{PeraltaAbadia2026thesis,
  author    = {José{-}Joaquín Peralta Abadía},
  title     = {Enhancing Smart Monitoring in Face Milling with Deep Continual Learning},
  school    = {Mondragon Unibertsitatea},
  year      = {2026}
}
```

---

## License

See [LICENSE](LICENSE).
