# RP-STDS-DTR: Radiation-Physics-Constrained Spatio-Temporal Dual-Stream Network for Dim Target Recognition in Deep-Space scenarios Remote Sensing

# RP-STDS-DTR

Code for the Radiation-Physics-Constrained Spatio-Temporal Dual-Stream Network for Dim Target Recognition in Remote Sensing.

RP-STDS-DTR combines radiative-intensity sequences with infrared images for three-class recognition. The temporal branch extracts multiscale radiative features and learns class-specific scale weights. The spatial branch uses a dilated ResNet-18 backbone, and temporal features guide spatial feature modulation before classification. Temporal-variation, spectral-concentration, and scale-weight regularization terms are included in training.

## Repository structure

```text
models/
  full_model.py          Spatio-temporal dual-stream network
  spatial_backbone.py    Dilated ResNet-18 spatial branch
losses.py                Classification and auxiliary loss terms
multimodal_data.py       Sequence–image data loading and group-based splitting
train_dual_stream.py     Training, validation, and test entry point
generate-data.py         Example physics-based simulation script
```

## Requirements

Python 3.10 is recommended. Install the required packages with:

```bash
pip install torch numpy pandas pillow openpyxl opencv-python scipy matplotlib
```

## Generate example data

Set `BASE_DIR` near the beginning of `generate-data.py` to the desired output directory, then run:

```bash
python generate-data.py
```

The script saves radiative-intensity data, infrared image sequences, visualizations, and image-quality measurements in separate `physics`, `images`, `visuals`, and `metrics` directories.

## Prepare a training manifest

Training uses a CSV manifest with one row for each candidate sequence. Each row pairs a radiative-intensity sequence with a corresponding candidate image (`frame_path`) or directory of candidate frames (`frames_dir`).

The required columns are `sequence_id`, `label`, and `radiation_path`, together with either `frame_path` or `frames_dir`. The optional `radiation_column` field selects a column from a CSV or Excel radiation file. Labels can be written as `target`, `decoy`, and `debris`, or as `0`, `1`, and `2`.

Samples derived from the same original sequence or trajectory should have the same `group_id`. The data loader keeps each group within a single training, validation, or test subset.

An example manifest row is:

```csv
sequence_id,group_id,label,radiation_path,radiation_column,frame_path
seq_0001_target,seq_0001,target,data/physics/0001.xlsx,Target_Radiation,data/candidates/seq_0001_target.jpg
```

Paths in the manifest may be absolute or relative to the manifest file.

## Train and evaluate

Run the following command from the repository root:

```bash
python train_dual_stream.py --manifest path/to/dataset_manifest.csv --epochs 50 --batch-size 16 --learning-rate 1e-4
```

The script divides complete sequence groups into training, validation, and test subsets. It selects the training checkpoint using validation weighted F1-score, evaluates temporal-scale configurations on the validation set, and tests both the full and selected deployment models.

By default, outputs are saved in `runs/rp_stds_dtr/`:

```text
best_model.pt
deployment_model.pt
history.json
test_metrics.json
```
