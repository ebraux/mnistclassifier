"""
This module defines a simple neural network for classifying MNIST digits using PyTorch.
Classes:
    MNISTClassifier: A neural network model for MNIST digit classification.
Dependencies:
    torch: PyTorch library for tensor computations and neural networks.
    torch.nn: PyTorch module containing neural network layers and functions.
    torch.optim: PyTorch module containing optimization algorithms.
    A simple neural network for classifying MNIST digits.
    Args:
        input_size (int): The size of the input layer. Default is 784 (28x28 pixels).
        hidden_size (int): The size of the hidden layer. Default is 128.
        num_classes (int): The number of output classes. Default is 10 (digits 0-9).
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.
    Attributes:
        model (nn.Sequential): The sequential model containing the layers of the neural network.
        criterion (nn.CrossEntropyLoss): The loss function used for training.
        optimizer (optim.Adam): The optimizer used for training.
        device (torch.device): The device on which the model is trained (CPU or GPU).
    Methods:
        forward(x):
            Defines the forward pass of the network. Flattens the input image and passes it through the network.
        train_model(train_loader, num_epochs=5):
            Trains the model using the provided training data loader for a specified number of epochs.
        evaluate_model(test_loader):
            Evaluates the model using the provided test data loader and returns the accuracy, predictions, and true labels.
        save_model(path='mnist_classifier.pth'):
            Saves the model weights to the specified file path.
        load_model(path='mnist_classifier.pth'):
            Loads the model weights from the specified file path.
        predict(x):
            Makes a prediction on new data.
"""
import torch
from torch import nn
from torch import optim

class MNISTClassifier(nn.Module):
    """
    A simple neural network for classifying MNIST digits.

    Args:
        input_size (int): The size of the input layer. Default is 784 (28x28 pixels).
        hidden_size (int): The size of the hidden layer. Default is 128.
        num_classes (int): The number of output classes. Default is 10 (digits 0-9).
        learning_rate (float): The learning rate for the optimizer. Default is 0.001.

    Attributes:
        model (nn.Sequential): The sequential model containing the layers of the neural network.
        criterion (nn.CrossEntropyLoss): The loss function used for training.
        optimizer (optim.Adam): The optimizer used for training.
        device (torch.device): The device on which the model is trained (CPU or GPU).

    Methods:
        forward(x):
            Defines the forward pass of the network. Flattens the input image and passes it through the network.
        train_model(train_loader, num_epochs=5):
            Trains the model using the provided training data loader for a specified number of epochs.
        evaluate_model(test_loader):
            Evaluates the model using the provided test data loader and returns the accuracy, predictions, and true labels.
        save_model(path='mnist_classifier.pth'):
            Saves the model weights to the specified file path.
        load_model(path='mnist_classifier.pth'):
            Loads the model weights from the specified file path.
        predict(x):
            Makes a prediction on new data.
    """

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, learning_rate=0.001):
        """
        Initializes the MNISTClassifier model.

        Args:
            input_size (int, optional): The size of the input layer. Default is 784.
            hidden_size (int, optional): The size of the hidden layer. Default is 128.
            num_classes (int, optional): The number of output classes. Default is 10.
            learning_rate (float, optional): The learning rate for the optimizer. Default is 0.001.

        Attributes:
            model (nn.Sequential): The neural network model consisting of input, hidden, and output layers.
            criterion (nn.CrossEntropyLoss): The loss function used for training.
            optimizer (optim.Adam): The optimizer used for training.
            device (torch.device): The device on which the model will be trained (CPU or GPU).
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

        # Initialisation des composants d'entraînement
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Déplacer le modèle sur GPU si disponible

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Parameters:
        x (torch.Tensor): The input tensor, typically a batch of images.

        Returns:
        torch.Tensor: The output tensor after passing through the model.
        """
        # Aplatir l'image d'entrée
        x = x.view(x.size(0), -1)
        return self.model(x)

    def train_model(self, train_loader, num_epochs=5):
        """
        Méthode d'entraînement du modèle.

        Args:
            train_loader (DataLoader): Le DataLoader contenant les données d'entraînement.
            num_epochs (int): Le nombre d'époques pour l'entraînement. Default est 5.

        Returns:
            list: Les pertes d'entraînement pour chaque époque.
        """
        self.train()  # Mettre le modèle en mode entraînement
        train_losses = []

        for epoch in range(num_epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                # Déplacer les données sur le même device que le modèle
                data, target = data.to(self.device), target.to(self.device)

                # Réinitialiser les gradients
                self.optimizer.zero_grad()

                # Propagation avant
                output = self(data)

                # Calcul de la perte
                loss = self.criterion(output, target)

                # Rétropropagation
                loss.backward()

                # Mise à jour des poids
                self.optimizer.step()

                # Statistiques
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                # Affichage de la progression
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], '
                          f'Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')

            # Statistiques de l'époque
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Accuracy: {epoch_acc:.2f}%')

        return train_losses

    def evaluate_model(self, test_loader):
        """
        Méthode d'évaluation du modèle.

        Args:
            test_loader (DataLoader): Le DataLoader contenant les données de test.

        Returns:
            tuple: L'exactitude du modèle, les prédictions et les étiquettes réelles.
        """
        self.eval()  # Mettre le modèle en mode évaluation
        correct = 0
        total = 0
        predictions = []
        true_labels = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                _, predicted = torch.max(output.data, 1)

                total += target.size(0)
                correct += (predicted == target).sum().item()

                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')

        return accuracy, predictions, true_labels

    def save_model(self, path='mnist_classifier.pth'):
        """
        Sauvegarder les poids du modèle.

        Args:
            path (str): Le chemin où sauvegarder les poids du modèle. Default est 'mnist_classifier.pth'.
        """
        torch.save(self.state_dict(), path)
        print(f"Modèle sauvegardé dans {path}")

    def load_model(self, path='mnist_classifier.pth'):
        """
        Charger les poids du modèle.

        Args:
            path (str): Le chemin d'où charger les poids du modèle. Default est 'mnist_classifier.pth'.
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Modèle chargé depuis {path}")

    def predict(self, x):
        """
        Faire une prédiction sur de nouvelles données.

        Args:
            x (Tensor): Les données d'entrée pour la prédiction.

        Returns:
            Tensor: Les prédictions du modèle.
        """
        # Mettre en mode évaluation
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self(x)
            _, predicted = torch.max(output.data, 1)
            return predicted