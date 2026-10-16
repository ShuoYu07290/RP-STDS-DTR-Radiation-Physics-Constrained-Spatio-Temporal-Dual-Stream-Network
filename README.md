# RP-STDS-DTR

Code accompanying *RP-STDS-DTR: Radiation-Physics-Constrained Spatio-Temporal Dual-Stream Network for Dim Target Recognition in Deep-Space Remote Sensing Scenarios*.

## Contents

| File | Description |
| --- | --- |
| `temporal_transformer.py` | Linear embedding, multiscale temporal windows, class-specific softmax scale weights, and temporal encoding. |
| `dilated_resnet.py` | Spatial ResNet-18 backbone with a stride-1, dilated fourth stage. |
| `model.py` | Temporal-guided spatial modulation, feature fusion, and three-class classification. |
| `loss.py` | Temporal-variation, spectral-concentration, and scale-weight regularization terms. |
| `train.py` | Training, validation, scale-selection, and evaluation workflow. |
| `simulate_data.py` | Example physics-based infrared sequence generator. |

The three classes are target, decoy, and debris. The temporal branch uses window sizes of 2, 4, 6, 8, 10, 12, 20, 24, 30, and 40, with 64-dimensional embedded features. The experiment settings use 10080-step radiative sequences; the spatial branch accepts single-channel 256 × 256 frames. The reference parameters in the loss are fixed at `gamma=0.15` and `delta=0.45` during training.

## Dependencies

- Python 3.10 or later
- PyTorch
- NumPy
- Pillow
- scikit-learn (for the metrics in `train.py`)

## Generate example sequences

From this directory, run:

```bash
python simulate_data.py --output example_data --samples-per-class 20 --seed 42
```

For each class and sample, the generator writes a normalized `radiation.npy` sequence and 100 grayscale PNG frames of size 256 × 256:

```text
example_data/
  target/
    sample_0000/
      radiation.npy
      frames/frame_00000.png
  decoy/
  debris/
```

The example follows the modeling chain from class-dependent micromotion and projected area through thermal balance and band-integrated radiation to small-target image formation. Its physical and sensor settings are representative values for illustrating that chain, rather than the calibrated settings used to construct the experimental datasets.

## Model and evaluation notes

`temporal_transformer.py` stores the class-specific scale coefficients in a trainable `C × K` matrix, initializes them from a standard normal distribution, and applies softmax over the retained scales. `model.py` combines the resulting temporal features with spatial features before producing three class scores. `loss.py` keeps `gamma` and `delta` fixed; it does not optimize them as model parameters.

`train.py` records the training settings and the evaluation definitions used in the study. It reports F1-score using fixed class-frequency weights pooled across the four groups, accuracy, and macro-averaged one-versus-rest precision, recall, specificity, false-alarm rate, and AUC. Provide the three pooled target, decoy, and debris counts through `--reference-class-counts`; the same counts are used for every group. Complete sequences are split before temporal windows are formed, so windows derived from one sequence remain in the same subset.

The training entry point refers to project-specific dataset, attention, residual-block, pruning, configuration, and checkpoint modules through the `rp_stds_dtr` imports shown in the source files. Those integrations are required to run the full training workflow.
