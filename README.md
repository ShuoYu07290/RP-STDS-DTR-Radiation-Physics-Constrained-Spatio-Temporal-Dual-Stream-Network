# RP-STDS-DTR: Radiation-Physics-Constrained Spatio-Temporal Dual-Stream Network for Dim Target Recognition in Deep-Space scenarios Remote Sensing

# RP-STDS-DTR

Code for *Radiation-Physics-Constrained Spatio-Temporal Dual-Stream Network for Dim Target Recognition in Remote Sensing*.

This repository currently provides five core implementation files and a basic physics-based simulation script. The complete implementation and full simulation code will be released after the paper is accepted.

## Files

| File | Description |
| --- | --- |
| `temporal_transformer.py` | Multiscale temporal feature extraction and class-specific adaptive scale weighting. |
| `dilated_resnet.py` | Spatial feature extraction using a dilated ResNet-18 backbone. |
| `model.py` | Temporal-guided spatial enhancement, feature fusion, and three-class classification. |
| `loss.py` | Temporal-variation and spectral-concentration constraints, together with scale-weight regularization. |
| `train.py` | Training, validation, scale selection, and evaluation procedures. |
| `simulate_data.py` | Basic example of physics-based radiative-sequence and infrared-image generation. |

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy
- Pillow
- scikit-learn

## Basic simulation example

Run the following command from this directory:

```bash
python simulate_data.py --output example_data --samples-per-class 20 --seed 42
```

For each generated sample, the script saves a normalized radiative-intensity sequence in `radiation.npy` and its corresponding infrared frames in the `frames` directory:

```text
example_data/
├── target/
│   └── sample_0000/
│       ├── radiation.npy
│       └── frames/
│           ├── frame_00000.png
│           └── ...
├── decoy/
└── debris/
```

The example follows the physical modeling sequence of micromotion, projected-area variation, thermal evolution, radiative-intensity variation, and infrared image formation. It uses representative parameters to illustrate the generation procedure.

## Model settings

The temporal stream uses window sizes of `[2, 4, 6, 8, 10, 12, 20, 24, 30, 40]` and a 64-dimensional feature embedding. Its class-specific scale coefficients form a trainable matrix and are normalized by softmax across temporal scales.

The spatial stream processes single-channel `256 × 256` images. Its fourth residual stage uses stride 1 and dilated convolutions to preserve spatial responses from dim targets. The network combines temporal and spatial features to produce classification scores for target, decoy, and debris.

In the physics-guided loss, the reference parameters are fixed at `gamma = 0.15` and `delta = 0.45`; they are not updated during training.

## Training and evaluation

`train.py` presents the training and evaluation workflow, including sequence-level data partitioning, model optimization, validation-based scale selection, and metric calculation. The reported metrics include weighted F1-score, accuracy, precision, recall, false-alarm rate, specificity, and AUC.

The training entry point uses project modules referenced by the `rp_stds_dtr` imports in the source files. The complete implementation and full simulation code will be added after the paper is accepted.
