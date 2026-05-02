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
        cmap='RdYlGn',
        vmin=0.0, 
        vmax=1.0, 
        linewidths=0.3, 
        linecolor='white',
        cbar_kws={'label': 'Accuracy'})
    ax.invert_yaxis()
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Layer Depth', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'Saved: {output_path}')

def main(args):
    df = pd.read_csv(args.csv_path)
    df = df.set_index('epoch')

    out_dir = os.path.dirname(os.path.abspath(args.csv_path))

    # digit_cols = [f'{layer}_digit' for layer in LAYER_ORDER]
    # color_cols = [f'{layer}_color' for layer in LAYER_ORDER]

    digit_cols = [f'{layer}_bird' for layer in LAYER_ORDER]
    color_cols = [f'{layer}_water' for layer in LAYER_ORDER]

    digit_df = df[digit_cols].rename(columns={f'{l}_digit': l for l in LAYER_ORDER})
    color_df = df[color_cols].rename(columns={f'{l}_color': l for l in LAYER_ORDER})

    plot_heatmap(digit_df, 'Digit Probe Accuracy by Layer and Epoch',
                 os.path.join(out_dir, f'digit_probe_heatmap_{args.regularization}.png'))
    plot_heatmap(color_df, 'Color Probe Accuracy by Layer and Epoch',
                 os.path.join(out_dir, f'color_probe_heatmap_{args.regularization}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help='Path to the probe_accuracy CSV from train.py')
    parser.add_argument('regularization', type=str, help="Please enter as the following: <reg_type>_<amount> ex: 'l2_1e-5'")
    args = parser.parse_args()
    main(args)
