# configs/

Hyperparameter configuration files for the experiments. Each file is a JSON dictionary of hyperparameters loaded directly by the training scripts.

## Files

| File | Loaded by | Description |
|---|---|---|
| `champion_model.json` | `train_ak.py` | Full HP config example |
| `cl_models.json` | `train_cl.py`, `train_separate.py` | Per-signal-group model names for CL. Maps each group (`all`, `AC`, `DC`, `ACDC`, `AC_table`, `DC_table`, `internals`) to the top-1 architecture from the LOCO-CV full model zoo (sw=500/ss=500/ds=2). All other HPs come from `champion_timefreq_domain.json`. |

## Notes

- The `cl_models.json` contains only `model_name`. All other HPs (learning rate, dropout, scalogram, etc.) must come from `champion_timefreq_domain.json`.
