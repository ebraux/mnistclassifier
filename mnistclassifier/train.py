import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def train():

    # Transformations des données
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Chargement des données
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

    # Préparation des loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # Création du modèle
    model = MNISTClassifier()
    
    # Entraînement
    train_losses = model.train_model(train_loader)


    
if __name__ == "__main__":
    train()
