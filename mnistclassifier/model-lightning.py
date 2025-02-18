import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class MNISTClassifier(pl.LightningModule):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Aplatir l'image
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

def prepare_mnist_data():
    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Télécharger et préparer le dataset
    mnist_data = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Diviser le dataset en train et validation
    train_size = int(0.8 * len(mnist_data))
    val_size = len(mnist_data) - train_size
    train_data, val_data = random_split(mnist_data, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)

    return train_loader, val_loader

def train_model():
    # Initialiser le modèle
    model = MNISTClassifier()

    # Initialiser le trainer
    trainer = pl.Trainer(
        max_epochs=10, 
        accelerator='auto', 
        devices=1
    )

    # Préparer les données
    train_loader, val_loader = prepare_mnist_data()

    # Entraîner le modèle
    trainer.fit(model, train_loader, val_loader)

    return model

# Point d'entrée principal
if __name__ == '__main__':
    trained_model = train_model()