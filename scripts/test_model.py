import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from detection_truque import CustomDataset, get_file_paths_and_labels, transform  # Importer depuis le fichier principal
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights  # Importer le modèle correct
import os

if torch.cuda.is_available():
    print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Training on CPU")

# Fonction mise à jour pour charger les chemins des fichiers et les labels
def get_file_paths_and_labels(directory):
    file_paths = []
    labels = []
    # Dictionnaire des classes avec leurs labels respectifs
    class_names = {'Authentic': 0, 'Manipulated': 1}  
    
    for class_name, label in class_names.items():
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            for fname in os.listdir(class_dir):
                if fname.endswith(('.tif', '.jpg', '.png')):
                    file_paths.append(os.path.join(class_dir, fname))
                    labels.append(label)
                    
    return file_paths, labels

# Créer et charger le modèle EfficientNet-B3
def create_model():
    # Charger le modèle pré-entraîné avec les poids ImageNet
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    model = efficientnet_b3(weights=weights)
    
    # Modifier la tête du modèle pour la classification binaire
    model.classifier = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model

# Charger le modèle et les poids
model = create_model()
model.load_state_dict(torch.load('efficientnetB3_weights_final.pth'))
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Fonction d'évaluation avec métriques
def evaluate_with_metrics(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = model(images).squeeze(1)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculer la matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Afficher la matrice sous forme de heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Truqué', 'Truqué'], yticklabels=['Non-Truqué', 'Truqué'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matrice de Confusion')
    plt.show()
    
    # Calculer les autres métriques
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Non-Truqué', 'Truqué']))

# Blocs principaux sous Windows
if __name__ == "__main__":
    # Charger les données de test
    test_paths, test_labels = get_file_paths_and_labels('data/truque/test')
    test_dataset = CustomDataset(test_paths, test_labels, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Évaluer le modèle sur le jeu de test
    evaluate_with_metrics(model, test_loader)
