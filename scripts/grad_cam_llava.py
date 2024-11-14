import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ollama
from torchvision.models import efficientnet_b3

# Charger le modèle EfficientNet-B3 avec les poids pré-entraînés
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Créer l'architecture du modèle
model = models.efficientnet_b3()
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 1)  # Sortie pour la classification binaire
)

# Charger les poids du modèle EfficientNet-B3 que tu as entraîné
model.load_state_dict(torch.load('efficientnetB3_weights_final.pth'))
model = model.to(device)
model.eval()  # Mode évaluation

# Fonction pour transformer les images (resize et normalisation)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fonction Grad-CAM pour générer la carte de chaleur
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        self.target_layer.register_forward_hook(self.save_gradient)
        self.target_layer.register_full_backward_hook(self.save_gradient_backprop)

    def save_gradient(self, module, input, output):
        self.gradients = output

    def save_gradient_backprop(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, inputs):
        outputs = self.model(inputs)
        self.model.zero_grad()
        target = torch.ones(outputs.size()).to(inputs.device)
        outputs.backward(gradient=target)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.gradients.detach()

        for i in range(activations.size(1)):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)  # ReLU activation
        heatmap = heatmap / np.max(heatmap)  # Normalisation
        return heatmap

# Fonction pour afficher la carte de chaleur sur l'image
def show_cam_on_image(img, heatmap):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    plt.imshow(cam_img)
    plt.show()

# Fonction pour générer un commentaire avec LLaVA
def generate_comment(truque):
    if truque:
        prompt = "Explique pourquoi cette image est truquée en analysant les anomalies visuelles détectées par le modèle."
    else:
        prompt = "Explique les défauts physiques visibles dans cette image qui semblent non truqués."
    
    response = ollama.generate(
        model="llava",  # Modèle LLaVA via Ollama
        prompt=prompt   # Prompt pour générer le commentaire
    )
    
    # Imprimer la réponse complète pour vérifier sa structure
    print("Réponse complète de LLaVA:", response)
    
    # Retourner le texte si disponible, sinon un message d'erreur
    return response.get('text', "Aucune réponse textuelle disponible")

# ----------- Main execution block ----------- #
if __name__ == "__main__":
    # Charger une image de test
    image_path = 'data/test/2.jpg'
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Appliquer le modèle pour obtenir la prédiction
    outputs = model(image_tensor).squeeze(1)
    prediction = torch.sigmoid(outputs).item()

    # Si l'image est détectée comme truquée (probabilité > 0.5)
    is_truque = prediction > 0.5

    # Appliquer Grad-CAM pour générer une carte de chaleur
    grad_cam = GradCAM(model=model, target_layer=model.features[-1])
    heatmap = grad_cam.generate_cam(image_tensor)

    # Visualiser la carte de chaleur sur l'image d'origine
    show_cam_on_image(image, heatmap)

    # Utiliser LLaVA pour générer un commentaire explicatif
    comment = generate_comment(is_truque)
    print(f"Commentaire généré : {comment}")
