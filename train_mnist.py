import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as tsfm
from torchvision.utils import make_grid
from collections import defaultdict
import matplotlib.pyplot as plt


class TwoLayerReLUNetwork(nn.Module):
    """
    GenerativeNetwork

    Using Kaiming Normal initialization.

    Parameters
    ----------
    k : int
        input dimension
    m : int
        width of hidden layer
    n : int
        output dimension
    """

    def __init__(self, k, m, n):
        super().__init__()
        self.input_dimension = k
        self.width = m
        self.output_dimension = n
        self.relu = nn.ReLU()

        self.first_layer = nn.Linear(k, m, bias=True)
        self.second_layer = nn.Linear(m, n, bias=True)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = self.relu(self.dropout(self.first_layer(x)))
        out = self.second_layer(out)
        return out


parms = {"k": 20, "m": 400, "n": 784}
num_epochs = 7

criterion = nn.MSELoss()
criterion_test = nn.MSELoss(reduction="sum")

network = TwoLayerReLUNetwork(**parms)
optimizer = optim.Adam(network.parameters())

image_transform = tsfm.Compose([tsfm.ToTensor(), lambda x: x.view(-1)])
label_map = torch.randn(10, parms["k"])


def label_transform(label):
    return label_map[label]


mnist_train = MNIST(
    "./data/",
    train=True,
    transform=image_transform,
    target_transform=label_transform,
    download=True,
)

mnist_test = MNIST(
    "./data/",
    train=False,
    transform=image_transform,
    target_transform=label_transform,
    download=True,
)
n_train = len(mnist_train)
n_test = len(mnist_test)


loader_train = DataLoader(mnist_train, batch_size=16, shuffle=True)
loader_test = DataLoader(mnist_test, batch_size=128, shuffle=False)


def train_step(batch):
    images, labels = batch
    optimizer.zero_grad()
    network.train()
    out = network(labels)
    loss = criterion(torch.clamp(out, 0, 1), images)
    loss.backward()
    optimizer.step()
    return loss.item()


def validate(loader):
    network.eval()
    eval_loss = 0.0
    for batch in loader:
        images, labels = batch
        with torch.no_grad():
            loss = criterion_test(torch.clamp(network(labels), 0, 1), images)
        eval_loss += loss.item()
    eval_loss /= n_test
    return eval_loss


def visualize():
    network.eval()
    with torch.no_grad():
        archetypes = torch.clamp(network(label_map).view(-1, 28, 28), 0, 1)
    fig, ax = plt.subplots(2, 5, figsize=(8, 6))
    ax = ax.ravel()
    for i in range(10):
        ax[i].imshow(archetypes[i])
    plt.show()
    plt.close("all")
    del fig, ax


def plot_history(history):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    ax[0].plot(history["train_loss"])
    ax[0].set_xlabel("iter")
    ax[0].set_ylabel("batch loss (mean normalized)")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[1].plot(history["eval_loss"])
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("validation loss (mean normalized)")
    plt.show()
    plt.close("all")
    del fig, ax


def save_first_layer(network):
    torch.save(network.first_layer.weight, "data/first_layer_mnist.pt")


if __name__ == "__main__":

    history = defaultdict(list)

    for epoch in range(num_epochs):
        for batch in loader_train:
            train_loss = train_step(batch)
            history["train_loss"].append(train_loss)
        eval_loss = validate(loader_test)
        history["eval_loss"].append(eval_loss)
        print(f"{epoch:02d}\t{eval_loss:.2e}")

    plot_history(history)
    visualize()

    save_first_layer(network)
