import torch
import torch.nn as nn
import torch.optim as optim


class MNISTClassifier(nn.Module):

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, learning_rate=0.001):
        super(MNISTClassifier, self).__init__()
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
        x = x.view(x.size(0), -1)
        return self.model(x)

    def train_model(self, train_loader, num_epochs=5):
        """
        Méthode d'entraînement du modele
        """
        self.train()  # Mettre le modele en mode entraînement
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
        Méthode d'évaluation du modèle
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
        Sauvegarder les poids du modèle
        """
        torch.save(self.state_dict(), path)
        print(f"Modèle sauvegardé dans {path}")

    def load_model(self, path='mnist_classifier.pth'):
        """
        Charger les poids du modèle
        """
        self.load_state_dict(torch.load(path))
        self.eval()
        print(f"Modèle chargé depuis {path}")

    def predict(self, x):
        """
        Faire une prédiction sur de nouvelles données
        """
        # Mettre en mode évaluation
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            output = self(x)
            _, predicted = torch.max(output.data, 1)
            return predicted
