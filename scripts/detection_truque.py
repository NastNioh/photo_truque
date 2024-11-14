import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torchvision.models import EfficientNet_B3_Weights

# Device configuration (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- STEP 1: Data Loading and Preprocessing ----------- #

# Fonction pour s'assurer qu'il y a exactement 3 canaux
def ensure_three_channels(x):
    return x[:3] if x.shape[0] > 3 else x

# Transformation des images (resize, data augmentation et normalisation)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(ensure_three_channels),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset personnalisé pour PyTorch
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx])

        if image.mode != 'RGB':
            image = image.convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]

        # Si les labels sont encodés en one-hot, convertir en un seul scalaire
        if isinstance(label, (list, np.ndarray)):
            label = np.argmax(label)

        return image, torch.tensor(label, dtype=torch.float32)

# Fonction pour récupérer les chemins des fichiers et les labels
def get_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    class_names = os.listdir(directory)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith(('.tif', '.jpg', '.png')):
                    file_paths.append(os.path.join(class_dir, fname))
                    labels.append(class_indices[class_name])
                    
    return file_paths, labels

# ----------- Main execution block ----------- #
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")
    
    base_dir = 'data/truque/train'
    all_paths, all_labels = get_file_paths_and_labels(base_dir)

    # Diviser en jeux d'entraînement et de validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(all_paths, all_labels, test_size=0.2, stratify=all_labels)

    # Calculer les poids des classes pour la pondération
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    
    # Obtenir uniquement le poids de la classe positive (classe 1 - "Truqué")
    pos_weight_tensor = torch.tensor(class_weights[1], dtype=torch.float32).to(device)

    # Créer les datasets et dataloaders
    train_dataset = CustomDataset(train_paths, train_labels, transform=transform)
    val_dataset = CustomDataset(val_paths, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # ----------- Model Definition (EfficientNet) ----------- #
    model = models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1)  # Pas de Sigmoid ici, car nous allons utiliser BCEWithLogitsLoss
    )
    model = model.to(device)

    # Define loss with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # Utilisation de BCEWithLogitsLoss
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    # ----------- Training Loop ----------- #
    def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
        train_losses, val_losses, val_accuracies = [], [], []
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images).squeeze(1)  # Sorties de la forme (batch_size,)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images).squeeze(1)  # Sorties de la forme (batch_size,)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(outputs) > 0.3).float()  # Appliquer Sigmoid ici
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            train_losses.append(running_loss/len(train_loader))
            val_losses.append(val_loss/len(val_loader))
            val_accuracies.append(correct / total)

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")

        return train_losses, val_losses, val_accuracies

    # Entraîner le modèle
    train_model(model, train_loader, val_loader, criterion, optimizer)

    # Sauvegarder les poids du modèle
    torch.save(model.state_dict(), 'efficientnetB3_weights_final.pth')
