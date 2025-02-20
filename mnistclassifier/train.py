"""
This module trains the MNIST classifier model.
"""

import torchvision.transforms as transforms
from datas from torchvision import transformsssifier

def train():
    """
    Entraîne le modèle de classification MNIST.

    Cette fonction effectue les étapes suivantes :
    1. Applique les transformations nécessaires aux données.
    2. Charge les données d'entraînement.
    3. Crée une instance du modèle MNISTClassifier.
    4. Entraîne le modèle avec les données d'entraînement.
    """
    # Transformations des données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données
    train_loader = load_data(transform, train=true)

    # Chargement des données de test
    # test_dataset =  load_data(transform, train=false)

    # # Préparation des loaders
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Création du modèle
    model = MNISTClassifier()

    # Entraînement
    model.train_model(train_loader)
    # train_losses = model.train_model(train_loader)


if __name__ == "__main__":
    train()
