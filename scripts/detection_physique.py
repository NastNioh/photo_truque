# import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
# from PIL import Image

# # Charger le processor et le modèle LLaVA
# processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
# model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").to("cuda" if torch.cuda.is_available() else "cpu")

# # Charger une image
# image_path = "data/defaut/1.jpeg"
# image = Image.open(image_path).convert("RGB")  # Assurez-vous que l'image est en RGB

# # Préparer le texte d'entrée et l'image
# input_text = "L'image a-t-elle des dommages physiques visibles ?"

# # Utiliser le processor pour préparer correctement les images et le texte
# inputs = processor(text=input_text, images=image, return_tensors="pt", padding=True).to(model.device)

# # Faire l'inférence
# with torch.no_grad():
#     outputs = model.generate(**inputs, max_new_tokens=100)  # max_new_tokens pour éviter le warning sur la longueur

# # Afficher les résultats
# print(processor.decode(outputs[0], skip_special_tokens=True))





# import torch
# from transformers import CLIPProcessor, CLIPModel
# from PIL import Image

# # Charger le modèle CLIP et le processor
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Charger l'image (remplacer par le chemin de l'image à tester)
# image_path = 'data/defaut/1.jpeg'
# image = Image.open(image_path)

# # Préparer le texte (descriptions possibles) et l'image pour le modèle
# descriptions = ["L'objet est endommagé", "L'objet est en bon état", "L'objet est légèrement endommagé"]
# inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)

# # Faire l'inférence
# with torch.no_grad():
#     outputs = model(**inputs)

# # Récupérer les scores de similarité texte-image
# logits_per_image = outputs.logits_per_image  # Score de similarité entre l'image et les descriptions
# probs = logits_per_image.softmax(dim=1)  # Convertir en probabilités

# # Afficher les probabilités pour chaque description
# for i, description in enumerate(descriptions):
#     print(f"Probabilité que {description.lower()} : {probs[0][i].item():.4f}")




# import requests

# # URL de l'API
# url = "http://localhost:10000/predict"

# # Fichier image
# image_path = "data/defaut/1.jpeg"
# prompt = "Décrivez l'image"

# # Préparer la requête
# with open(image_path, "rb") as image_file:
#     files = {"image": image_file}
#     data = {"prompt": prompt}
#     response = requests.post(url, files=files, data=data)

# # Afficher la réponse
# print(response.json())



import os
from PIL import Image
from io import BytesIO
import torch
from LLaVA.predict import Predictor # Assurez-vous que le chemin vers votre fichier est correct

# Chemin de l'image dans votre dossier local
image_path = "data/defaut/1.jpeg.jpg"  # Changez ceci pour le chemin de votre image locale

# Prompt pour la génération de texte
prompt = "Describe the image"

# Initialiser le prédicteur
predictor = Predictor()
predictor.setup()

# Charger l'image à partir du chemin
image = load_image(image_path)

# Effectuer la prédiction
result = predictor.predict(image=image_path, prompt=prompt)

# Afficher les résultats
print("Résultats de la prédiction :")
for line in result:
    print(line)

