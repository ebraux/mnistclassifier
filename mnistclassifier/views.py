
"""
This module contains functions for visualizing and analyzing datasets and models.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset(dataset, num_samples=10):
    """
    Visualisation des Datasets
    Visualisation d'un échantillon aléatoire d'images du dataset
    Montre l'image et son label
    """
    plt.figure(figsize=(15, 3))

    # Sélectionner un échantillon aléatoire
    indices = np.random.randint(0, len(dataset), num_samples)

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # Convertir le tenseur en image numpy
        img = image.squeeze().numpy()

        plt.subplot(1, num_samples, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'Label: {label}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def analyze_dataset_distribution(dataset):
    """
    Distribution des classes
    Crée un graphique à barres montrant la distribution des classes
    Permet de vérifier l'équilibre du dataset
    """
    labels = [label for _, label in dataset]
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts)
    plt.title('Distribution des Classes')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'échantillons')
    plt.xticks(unique)

    # Afficher le nombre exact
    for i, count in enumerate(counts):
        plt.text(unique[i], count, str(count), ha='center', va='bottom')

    plt.show()


def detailed_model_evaluation(model, test_loader):
    """
    Analyse détaillée
    Évaluation détaillée avec matrice de confusion et exemples de prédictions
    - Crée une matrice de confusion
    - Visualise les images mal classées
    - Montre où le modèle fait des erreurs
    La matrice de confusion montre où le modèle confond des classes
    Les images mal classées aident à comprendre les limites du modèle
    """
    model.eval()

    # Matrices pour stocker les résultats
    confusion_matrix = np.zeros((10, 10), dtype=int)
    misclassified_images = []

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)

            # Matrice de confusion
            for t, p in zip(target, predicted):
                confusion_matrix[t.item(), p.item()] += 1

            # Collecter les images mal classées
            mask = predicted != target
            for img, true_label, pred_label in zip(data[mask], target[mask], predicted[mask]):
                misclassified_images.append((img, true_label.item(), pred_label.item()))

    # Visualiser la matrice de confusion
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    plt.title('Matrice de Confusion')
    plt.colorbar()
    plt.xlabel('Classe Prédite')
    plt.ylabel('Classe Réelle')

    # Ajouter les valeurs dans la matrice
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(confusion_matrix[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center")

    plt.tight_layout()
    plt.show()
