"""
Generate probe accuracy heatmaps from training CSV output.

Auto-detects the dataset from the CSV column names (_bird/_water vs _color/_digit).

Examples:
    python plot_heatmap.py results/waterbirds/l1/05-11-probe_accuracy_l1_1e-3.csv l1_1e-3
    python plot_heatmap.py results/colored_mnist/l2/05-10-probe_accuracy_l2_1e-3.csv l2_1e-3
"""

import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LAYER_ORDER = ['relu', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']


def plot_heatmap(data, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        data.T,
        ax=ax,
        annot=True,
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        linewidths=0.3,
        linecolor='white',
        cbar_kws={'label': 'Accuracy'},
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Layer Depth', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved: {output_path}')


def detect_dataset(columns):
    """Detect dataset from CSV column names."""
    col_str = ' '.join(columns)
    if '_bird' in col_str or '_water' in col_str:
        return 'waterbirds'
    elif '_color' in col_str or '_digit' in col_str:
        return 'cmnist'
    else:
        raise ValueError(f"Cannot detect dataset from columns: {list(columns)}")


def main(args):
    df = pd.read_csv(args.csv_path)
    df = df.set_index('epoch')

    out_dir = os.path.dirname(os.path.abspath(args.csv_path))
    dataset = detect_dataset(df.columns)

    if dataset == 'waterbirds':
        spurious_name, core_name = 'water', 'bird'
        spurious_title = 'Water (Background) Probe Accuracy'
        core_title = 'Bird Probe Accuracy'
    else:
        spurious_name, core_name = 'color', 'digit'
        spurious_title = 'Color Probe Accuracy'
        core_title = 'Digit Probe Accuracy'

    spurious_cols = [f'{layer}_{spurious_name}' for layer in LAYER_ORDER]
    core_cols = [f'{layer}_{core_name}' for layer in LAYER_ORDER]

    spurious_df = df[spurious_cols].rename(
        columns={f'{l}_{spurious_name}': l for l in LAYER_ORDER}
    )
    core_df = df[core_cols].rename(
        columns={f'{l}_{core_name}': l for l in LAYER_ORDER}
    )

    plot_heatmap(
        core_df,
        f'{core_title} by Layer and Epoch',
        os.path.join(out_dir, f'{core_name}_probe_heatmap_{args.regularization}.png'),
    )
    plot_heatmap(
        spurious_df,
        f'{spurious_title} by Layer and Epoch',
        os.path.join(out_dir, f'{spurious_name}_probe_heatmap_{args.regularization}.png'),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate probe accuracy heatmaps")
    parser.add_argument('csv_path', type=str,
                        help='Path to the probe_accuracy CSV from train.py')
    parser.add_argument('regularization', type=str,
                        help='Label for output filenames, e.g. l2_1e-3 or dropout_0.3')
    args = parser.parse_args()
    main(args)
