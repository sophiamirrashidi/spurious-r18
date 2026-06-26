import csv
import os


def log_epoch_accuracy(epoch, epoch_correct, epoch_total, csv_path):
    """Append one row of probe accuracies to the CSV log."""
    row = {'epoch': epoch}
    for name, correct in epoch_correct.items():
        row[name] = correct / epoch_total

    write_header = not os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
