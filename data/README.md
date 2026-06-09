# data/

Data loading and preprocessing for both the NASA Ames Milling Dataset and the MU-TCM dataset. On first training run the loaders build a binary cache (e.g., `data/nasa.bin`) so subsequent runs are fast. The cache files are not committed to the repository — they are regenerated automatically.

## Files

| File | Description |
|---|---|
| `mat_to_csv.py` | One-time conversion of the raw NASA `.mat` file (`data/mill.mat`) to a structured CSV. Run this before the first training run if the raw MAT file is available but the CSV is not. |
| `datasets.py` | `NASA_Dataset` — PyTorch `Dataset` subclass for the NASA milling data. Handles windowing, downsampling, z-score normalisation (fit on training set), signal group selection, LOCO-CV and paper splits, and binary cache management. Entry point: `get_nasa_data_pipeline()`. |
| `mu_tcm_datasets.py` | `MU-TCM` data loader and scenario construction. Builds experience streams for Avalanche CL (Mode A single-condition holdout / Mode B material-level holdout). Entry point: `get_mu_tcm_scenario_data()`. |
| `transforms.py` | Scalogram and normalisation transforms applied per window: `StdScalerTransform`, `MinMaxScalerTransform`, `StdScaler3D`, STFT magnitude, CWT, WPD, FSST. Used internally by `datasets.py`. |

## Data setup

### NASA dataset

1. Download `mill.mat` from [NASA Ames Prognostics Data Repository](https://data.nasa.gov/dataset/milling-wear).
2. Place it at `data/mill.mat`.
3. Convert to CSV (needed only once):
   ```bash
   python data/mat_to_csv.py
   ```
4. The binary cache `data/nasa.bin` is built automatically when `datasets.py` is first imported with the CSV present.

### MU-TCM dataset

1. Download from [https://hdl.handle.net/20.500.11984/6926](https://hdl.handle.net/20.500.11984/6926) (Peralta et al., 2025).
2. Place the dataset directory at `data/MU-TCM/`.
3. The cache `data/mu.bin` is built automatically on first use.

## Key parameters

All windowing parameters are set in the training scripts, not in this module. The champion configuration uses:
- Window size: **sw = 500** post-downsampling samples
- Stride: **ss = 500** (non-overlapping)
- Downsample factor: **ds = 2**
- Effective temporal coverage per window: **1000 raw samples = 2 s @ 500 Hz (NASA) / 4 s @ 250 Hz (MU-TCM)**

## Fixed train/test split (NASA)

The paper split is hard-coded and must never be changed:
- **Train cases:** {1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14} (86 runs)
- **Test cases:** {11, 12, 15, 16} (37 runs)
- **Case 6 is always excluded** (documented anomaly).

This split was established by Zheng et al. (2017) and is used for all cross-condition evaluation reported in the thesis.
