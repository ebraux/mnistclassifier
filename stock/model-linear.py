import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datas import load_data


def main():
    # Définition des transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données d'entraînement
    train_loader = load_data(transform, train=true)


    # Initialisation du modèle
    model = MNISTClassifier()

    # Définition de la fonction de perte et de l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entraînement du modèle
    train_model(model, train_loader, criterion, optimizer)

    # Évaluation du modèle
    evaluate_model(model, test_loader)

    # Sauvegarde du modèle
    torch.save(model.state_dict(), 'mnist_classifier.pth')


if __name__ == '__main__':
    main()
