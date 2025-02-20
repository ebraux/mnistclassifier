"""
This module demonstrates the prediction capabilities of a given model on the MNIST dataset.
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def demo_prediction(model):
    """
    Demonstrates the prediction capabilities of a given model on the MNIST dataset.

    Args:
        model: The trained model used for making predictions.

    Returns:
        tuple: A tuple containing:
            - images (Tensor): The batch of images used for the demonstration.
            - predictions (Tensor): The predicted labels for the batch of images.
            - true_labels (Tensor): The true labels for the batch of images.

    The function performs the following steps:
        1. Configures the necessary transformations for the MNIST dataset.
        2. Loads a small batch of test data from the MNIST dataset.
        3. Uses the model to make predictions on the batch of test data.
        4. Visualizes the images along with their predicted and true labels.
        5. Prints the accuracy of the model on the sample batch.
    """
    # Configuration
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    
    # Charger quelques données de test
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Créer un petit DataLoader pour la démonstration
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Obtenir un batch de données pour la démonstration
    images, true_labels = next(iter(test_loader))

    # Faire des prédictions
    predictions = model.predict(images)

    # Visualiser les résultats
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i].squeeze().numpy(), cmap='gray')
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        plt.title(f'Pred: {predictions[i]}\nVrai: {true_labels[i]}',
                  color=color,
                  fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    #  Afficher quelques statistiques
    correct = (predictions == true_labels).sum().item()
    total = len(predictions)
    print(f"Précision sur l'échantillon: {100 * correct / total:.2f}%")
    return images, predictions, true_labels


def demo_individual_prediction(model, index=None):
    """
    Faire une prédiction sur une seule image du jeu de test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # -- data.py--
    # # Charger le dataset de test
    # test_dataset = torchvision.datasets.MNIST(
    #     root='./data',
    #     train=False,
    #     download=True,
    #     transform=transform
    # )

    # Si aucun index n'est spécifié, en choisir un au hasard
    if index is None:
        index = torch.randint(0, len(test_dataset), (1,)).item()

    # Obtenir une image
    image, true_label = test_dataset[index]

    # Faire une prédiction
    prediction = model.predict(image.unsqueeze(0))

    # Visualiser l'image et la prédiction
    plt.figure(figsize=(3, 3))
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    color = 'green' if prediction.item() == true_label else 'red'
    plt.title(f'Prédiction: {prediction.item()}\nVrai label: {true_label}',
              color=color)
    plt.axis('off')
    plt.show()

    return prediction.item(), true_label
