import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from kymatio import Scattering2D
from matplotlib import cm
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import project1
from project1.ConvNetMods import alexnetmod, vgg16mod, resnetmod
from project1.Dataset import TransformedMNIST
from project1.FeatureExtractor import FeatureExtractor

project1.ConvNetMods.alexnetmod

def visualize_tsne(dataloader):
    batch_id, [features, labels] = next(enumerate(dataloader))
    tsne = TSNE(n_components=2, perplexity=30.0, n_iter=50000, init="pca")
    plot_only = 500
    embeddings = tsne.fit_transform(features.numpy()[:plot_only, :])
    labels = labels[:plot_only]
    plot_with_labels(embeddings, labels, batch_id)


# visualize some features using tsne
def plot_with_labels(weights, labels, batch_id):
    plt.cla()
    if type(labels[0]) == torch.Tensor:
        labels_plt = [x.item() for x in labels]
    else:
        labels_plt = labels

    X, Y = weights[:, 0], weights[:, 1]
    for x, y, s in zip(X, Y, labels_plt):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)

    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title("Visualize features {}".format(batch_id))


def imshow(inp, title=None, normalize=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, interpolation="bilinear", aspect="auto")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def extract_mnist_features(ignore=[], save_to_disk=True, train=True):
    models = []
    if "alexnet" not in ignore:
        alexnet = alexnetmod()
        models.append(alexnet)

    if "vgg16" not in ignore:
        vgg16 = vgg16mod()
        models.append(vgg16)

    if "resnet" not in ignore:
        resnet = resnetmod()
        models.append(resnet)
    # get dataset
    mnist = TransformedMNIST()
    batch_size = {"alexnetmod": 1000, "vgg16mod": 100, "resnetmod": 100}
    for model in models:
        name = model.__class__.__name__
        extractor = FeatureExtractor(model)
        dataset = mnist.get_train() if train else mnist.get_test()
        dataloader = DataLoader(dataset, batch_size=batch_size[name], num_workers=2)

        _ = extractor.features(dataloader, save_to_disk=save_to_disk, train=train)


def scattering_transform_mnist(save_to_disk=True, train=True):
    # here we want untransformed mnist data
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(os.getcwd() + "/mnist",
                                 train=True,
                                 transform=transform,
                                 download=True)
    mnist_test = datasets.MNIST(os.getcwd() + "/mnist",
                                train=False,
                                transform=transform,
                                download=True)

    # construct the scattering object
    scattering = Scattering2D(J=2, shape=(28, 28))
    dataloader = DataLoader(mnist_train if train else mnist_test, batch_size=1000)

    print("Running scattering transform")
    extractor = FeatureExtractor(scattering)
    out_features, out_labels = extractor.features(dataloader, save_to_disk=save_to_disk, train=train)
