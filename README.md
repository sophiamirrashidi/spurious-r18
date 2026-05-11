# spurious-r18

Experiments studying how spurious correlations are encoded in ResNet18, and whether regularization reduces reliance on them. Linear probes are attached at each layer (relu, layer1–4, avgpool) and trained simultaneously with the main model to measure what features each layer encodes over time.

Two datasets are used — **Colored MNIST** (digit-color correlation) and **Waterbirds** (bird-background correlation) — across three regularization strategies: L1, L2 (weight decay), and Dropout.

## Datasets

### Colored MNIST

Generate the dataset before training:

```bash
python generate_colored_mnist.py
```

This creates a `data/` directory with train/test splits where digit color is correlated with the digit label.

### Waterbirds

Download the Waterbirds dataset and place it at `../waterbird_complete95_forest2water2` relative to this repo (the default `--datapath`). You can override this with `--datapath /your/path`.

---

## Training Experiments

### Colored MNIST

**L1 regularization** — use `train.py` with `--l1_lambda`:
```bash
python train.py --l1_lambda 1e-3 --num_epochs 40 --save_path resnet_l1.pt
```

**L2 regularization (weight decay)** — use `train.py` with `--weight_decay`:
```bash
python train.py --weight_decay 1e-3 --num_epochs 40
```

**Dropout** — use `train_dropout.py` with `--dropout_rate`:
```bash
python train_dropout.py --dropout_rate 0.3 --num_epochs 40 --save_path resnet_dropout.pt
```

### Waterbirds

**L1 regularization** — use `train_waterbirds_L1.py`:
```bash
python train_waterbirds_L1.py --l1_lambda 1e-3 --datapath ../waterbird_complete95_forest2water2 --save_path resnet_wb_l1.pt
```

**L2 regularization** — use `train_waterbirds.py` with `--weight_decay`:
```bash
python train_waterbirds.py --weight_decay 1e-3 --datapath ../waterbird_complete95_forest2water2
```

**Dropout** — use `train_waterbirds_dropout.py`:
```bash
python train_waterbirds_dropout.py --dropout_rate 0.3 --datapath ../waterbird_complete95_forest2water2
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--l1_lambda` | `1e-5` | L1 regularization weight |
| `--weight_decay` | `None` / `1e-5` | L2 weight decay passed to SGD optimizer |
| `--dropout_rate` | `0.0` | Dropout probability inserted after each residual block |
| `--num_epochs` | `40` (10 for WB dropout) | Training epochs |
| `--lr` | `0.01` | Model learning rate |
| `--save_path` | `./resnet.pt` | Where to save the trained model weights |
| `--datapath` | `../waterbird_complete95_forest2water2` | Path to Waterbirds dataset (WB scripts only) |

---

## Outputs

Each training run produces:

- A `.pt` model checkpoint (path set by `--save_path`)
- A CSV of probe accuracies per epoch, saved alongside the model (e.g. `wb_l1/probe_accuracies_l1_1e-3.csv`)

The CSV has one row per epoch with columns for each layer × attribute combination, e.g. `relu_bird`, `layer1_water`, `avgpool_bird`, etc.

---

## Generating Heatmaps

After training, visualize probe accuracy across layers and epochs using `plot_heatmap.py`. This script takes the probe accuracy CSV and a label for the regularization configuration:

```bash
python plot_heatmap.py <path/to/probe_accuracies.csv> <regularization_label>
```

The `regularization_label` is used only for naming the output PNG files. Use the format `<type>_<value>`, e.g. `l1_1e-3` or `dropout_0.3`.

**Example — Waterbirds L1 run:**
```bash
python plot_heatmap.py wb_l1/probe_accuracies_l1_1e-3.csv l1_1e-3
```

This saves two heatmaps in the same directory as the CSV:
- `bird_probe_heatmap_l1_1e-3.png` — bird-type probe accuracy by layer and epoch
- `water_probe_heatmap_l1_1e-3.png` — water-background probe accuracy by layer and epoch

> **Note:** `plot_heatmap.py` is hardcoded for Waterbirds probe columns (`_bird` / `_water`). To use it with Colored MNIST CSVs (which have `_color` / `_digit` columns), manually update the column name variables near the top of `plot_heatmap.py`:
>
> ```python
> # Change these two lines:
> bird_cols = [f'{layer}_bird' for layer in LAYER_ORDER]
> water_cols = [f'{layer}_water' for layer in LAYER_ORDER]
>
> # To:
> bird_cols = [f'{layer}_color' for layer in LAYER_ORDER]
> water_cols = [f'{layer}_digit' for layer in LAYER_ORDER]
> ```
>
> Also update the `plot_heatmap(...)` title strings and output filenames on the lines below to match.
