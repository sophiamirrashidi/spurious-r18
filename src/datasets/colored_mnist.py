import torch
from torch.utils.data import Dataset
from torchvision import datasets


def make_colored_mnist(images, labels, color_correlation=0.9):
    """
    Build colored MNIST per the IRM paper (Arjovsky et al., 2019).
    """
    # binarize
    binary_labels = (labels >= 5).long()

    # flip label with p=0.25 -- makes digit shape noisier to predict label
    flip = torch.bernoulli(torch.full_like(binary_labels.float(), 0.25)).long()
    noisy_labels = (binary_labels + flip) % 2

    # assign color
    color_noise = torch.bernoulli(
        torch.full_like(noisy_labels.float(), 1 - color_correlation)
    ).long()
    colors = (noisy_labels + color_noise) % 2  # 0=red, 1=green

    # build 3-channel RGB image
    imgs = images.float() / 255.0
    colored = torch.zeros(len(imgs), 3, 28, 28, dtype=torch.float32)
    for i in range(len(imgs)):
        colored[i, colors[i].item()] = imgs[i]

    return colored, binary_labels, colors


class ColoredMNIST(Dataset):
    """
    Colored MNIST dataset with a fixed color-label correlation.
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
        split_seeds = {'train': 0, 'val': 1, 'test': 2}
        rng_state = torch.get_rng_state()
        torch.manual_seed(split_seeds[split])
        self.images, self.digit_labels, self.color_labels = make_colored_mnist(
            images, labels, color_correlation
        )
        torch.set_rng_state(rng_state)

    def __len__(self):
        return len(self.digit_labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.digit_labels[idx], self.color_labels[idx]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    train_dataset = ColoredMNIST(root='~/datasets/mnist', split='train', color_correlation=0.9)
    val_dataset = ColoredMNIST(root='~/datasets/mnist', split='val', color_correlation=0.7)
    test_dataset = ColoredMNIST(root='~/datasets/mnist', split='test', color_correlation=0.5)
    plt.imshow(train_dataset[1][0].permute(1, 2, 0))
    plt.show()
    print(f'Train size: {len(train_dataset)}')
    print(f'Val size:   {len(val_dataset)}')
    print(f'Test size:  {len(test_dataset)}')
    print(f'Image shape: {train_dataset[0][0].shape}')
    print(f'Label: {train_dataset[0][1]}')
