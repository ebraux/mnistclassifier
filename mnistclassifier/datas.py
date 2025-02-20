"""
This module provides functions to load the MNIST dataset using torchvision.
"""

import torchvision
from torch.utils.data import DataLoader


def load_data(transform, batch_size=64, shuffle=True, train=True):
    """
    Charge les données MNIST.

    Args:
        transform: Les transformations à appliquer aux données.
        batch_size: La taille des lots de données.
        shuffle: Indique si les données doivent être mélangées.
        train: Indique si les données d'entraînement ou de test doivent être chargées.

    Returns:
        DataLoader: Un DataLoader pour les données MNIST.
    """
    dataset = torchvision.datasets.MNIST(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

# Exemple d'utilisation
# transform = ...  # Définir les transformations ici
# train_loader = load_data(transform, train=True)
# test_loader = load_data(transform, train=False)
