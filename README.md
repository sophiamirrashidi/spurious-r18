# spurious-r18

Experiments studying how spurious correlations are encoded across ResNet18 layers and whether regularization reduces reliance on them.

Linear probes are attached at each layer (`relu`, `layer1`–`layer4`, `avgpool`) and trained simultaneously with the main model to measure what features each layer encodes over training time.

## Datasets

- **Colored MNIST** — digit-color correlation (binary: digit ≥ 5, color = red/green)
- **Waterbirds** — bird type correlated with background (land vs. water)

## Project Structure

```
├── train.py              # Unified training script (all datasets + regularizers)
├── eval.py               # Model evaluation
├── plot_heatmap.py       # Probe accuracy heatmap visualization
│
├── src/                  # Shared library code
│   ├── models.py         # ResNet18 variants (CMNIST, Waterbirds, Dropout wrapper)
│   ├── probes.py         # Linear probe definition and forward pass
│   ├── regularization.py # L1 regularization
│   ├── logging_utils.py  # CSV epoch logging
│   └── datasets/         # Dataset classes
│       ├── colored_mnist.py
│       └── waterbirds.py
│
├── experiments/          # Shell scripts for running full sweeps
│   ├── train_cmnist.sh
│   └── train_waterbirds.sh
│
└── results/              # Training outputs (CSVs + heatmap PNGs)
    ├── colored_mnist/
    │   ├── l1/
    │   ├── l2/
    │   └── dropout/
    └── waterbirds/
        ├── baseline/
        ├── l1/
        ├── l2/
        └── dropout/
```

## Setup

Requires Python ≥ 3.11. Install dependencies:

```bash
pip install -e .
```

### Colored MNIST

Data is generated on the fly from MNIST (downloaded automatically to `~/datasets/mnist` by default).

### Waterbirds

Download the Waterbirds dataset and place it at `../waterbird_complete95_forest2water2` (or specify with `--datapath`).

---

## Training

A single `train.py` handles all experiment configurations:

```bash
# Colored MNIST with L2 regularization
python train.py --dataset cmnist --reg l2 --weight_decay 1e-3

# Colored MNIST with L1 regularization
python train.py --dataset cmnist --reg l1 --l1_lambda 1e-3

# Colored MNIST with Dropout
python train.py --dataset cmnist --reg dropout --dropout_rate 0.3

# Waterbirds baseline (no regularization)
python train.py --dataset waterbirds --reg none

# Waterbirds with L1
python train.py --dataset waterbirds --reg l1 --l1_lambda 1e-3 --datapath ../waterbird_complete95_forest2water2
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | (required) | `cmnist` or `waterbirds` |
| `--reg` | `none` | Regularization: `none`, `l1`, `l2`, `dropout` |
| `--l1_lambda` | `1e-5` | L1 regularization weight |
| `--weight_decay` | `1e-3` | L2 weight decay for SGD |
| `--dropout_rate` | `0.3` | Dropout probability between residual blocks |
| `--num_epochs` | `40` | Training epochs |
| `--lr` | `0.01` | Model learning rate |
| `--probe_lr` | `0.01` | Probe learning rate |
| `--output_dir` | `results/<dataset>/<reg>` | Where to save outputs |
| `--data_root` | `~/datasets/mnist` | MNIST data root (cmnist only) |
| `--datapath` | `../waterbird_complete95_forest2water2` | Waterbirds path |

### Running Full Sweeps

```bash
bash experiments/train_cmnist.sh
bash experiments/train_waterbirds.sh
```

---

## Outputs

Each training run produces:
- A `.pt` model checkpoint in the output directory
- A CSV of probe accuracies per epoch (one row per epoch, columns: `<layer>_<attribute>`)

Column naming: `relu_color`, `layer1_digit`, `avgpool_bird`, `layer3_water`, etc.

---

## Generating Heatmaps

Visualize probe accuracy across layers and epochs:

```bash
python plot_heatmap.py <path/to/probe_accuracy.csv> <regularization_label>
```

The script auto-detects the dataset from column names. The regularization label is used for output filenames.

**Examples:**

```bash
python plot_heatmap.py results/waterbirds/l1/05-05-17-38_wb_probe_accuracy_L1_le-5.csv l1_1e-5
python plot_heatmap.py results/colored_mnist/l2/04-18-10-24_probe_accuracy_l2_reg_1e-05.csv l2_1e-5
```

---

## Evaluation

```bash
python eval.py --dataset cmnist --model_path results/colored_mnist/l2/resnet_l2_1e-3.pt
python eval.py --dataset waterbirds --model_path results/waterbirds/baseline/resnet_baseline.pt --datapath ../waterbird_complete95_forest2water2
```

For dropout models, pass `--dropout_rate` matching training:

```bash
python eval.py --dataset cmnist --model_path results/colored_mnist/dropout/resnet_dropout_0.3.pt --dropout_rate 0.3
```
