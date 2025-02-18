import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Définition du modèle de réseau de neurones
class MNISTClassifier(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(MNISTClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        # Aplatir l'image d'entrée
        x = x.view(x.size(0), -1)
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, num_epochs=5):
    """Fonction d'entraînement du modèle"""
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Réinitialiser les gradients
            optimizer.zero_grad()
            
            # Propagation avant
            output = model(data)
            
            # Calcul de la perte
            loss = criterion(output, target)
            
            # Rétropropagation
            loss.backward()
            
            # Mise à jour des poids
            optimizer.step()
            
            total_loss += loss.item()
        
        # Affichage des statistiques par époque
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

def evaluate_model(model, test_loader):
    """Fonction d'évaluation du modèle"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    # Définition des transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données d'entraînement
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Chargement des données de test
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

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

    