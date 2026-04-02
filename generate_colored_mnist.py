import torch
from torch.utils.data import Dataset
from torchvision import datasets


def make_colored_mnist(images, labels, color_correlation=0.9):
    """
    Build colored MNIST per the IRM paper (Arjovsky et al., 2019).

    Steps:
      1. Binarize digit label: y = 1 if digit >= 5, else y = 0
      2. Flip label with p=0.25 to make shape an imperfect predictor
      3. Assign color correlated with noisy label at rate `color_correlation`:
           c = noisy_label  with prob color_correlation
           c = 1-noisy_label with prob 1-color_correlation
      4. Color image red (c=0) or green (c=1)

    Returns:
      colored_images: float32 tensor of shape (N, 2, 28, 28), values in [0, 1]
      binary_labels:  long tensor of shape (N,), values in {0, 1}
    """
    # Step 1: binarize
    binary_labels = (labels >= 5).long()

    # Step 2: flip label with p=0.25 -- makes digit shape noisier to predict label
    flip = torch.bernoulli(torch.full_like(binary_labels.float(), 0.25)).long()
    noisy_labels = (binary_labels + flip) % 2

    # Step 3: assign color
    color_noise = torch.bernoulli(
        torch.full_like(noisy_labels.float(), 1 - color_correlation)
    ).long()
    colors = (noisy_labels + color_noise) % 2  # 0=red, 1=green

    # Step 4: build 3-channel RGB image
    # Red image  -> (pixel, 0, 0), green image -> (0, pixel, 0), blue always 0
    # images: (N, 28, 28), uint8
    imgs = images.float() / 255.0  # (N, 28, 28)
    colored = torch.zeros(len(imgs), 3, 28, 28, dtype=torch.float32)
    for i in range(len(imgs)):
        colored[i, colors[i].item()] = imgs[i]

    return colored, binary_labels


class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset with a fixed color-label correlation.

    Args:
        root:              path for downloading raw MNIST
        split:             'train' (first 50k) or 'val' (last 10k)
        color_correlation: probability that color matches the (noisy) label
        transform:         optional transform applied to each image tensor
    """

    def __init__(self, root, split='train', color_correlation=0.9, transform=None):
        assert split in ('train', 'val', 'test')
        self.transform = transform

        if split == 'test':
            mnist = datasets.MNIST(root, train=False, download=True)
            images, labels = mnist.data, mnist.targets
        else:
            mnist = datasets.MNIST(root, train=True, download=True)
            if split == 'train':
                images, labels = mnist.data[:50000], mnist.targets[:50000]
            else:
                images, labels = mnist.data[50000:], mnist.targets[50000:]

        self.images, self.labels = make_colored_mnist(images, labels, color_correlation)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  # (2, 28, 28)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]


if __name__ == '__main__':
    train_dataset = ColoredMNIST(root='~/datasets/mnist', split='train', color_correlation=0.9)
    val_dataset   = ColoredMNIST(root='~/datasets/mnist', split='val',   color_correlation=0.7)
    test_dataset  = ColoredMNIST(root='~/datasets/mnist', split='test',  color_correlation=0.5)

    print(f'Train size: {len(train_dataset)}')
    print(f'Val size:   {len(val_dataset)}')
    print(f'Test size:  {len(test_dataset)}')
    print(f'Image shape: {train_dataset[0][0].shape}')
    print(f'Label: {train_dataset[0][1]}')
