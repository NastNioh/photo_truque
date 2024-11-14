# import os
# from fastapi import FastAPI, UploadFile, File, Form
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import ollama
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from scripts.grad_cam import GradCAM
# from scripts.rules import automated_decision
# from scripts.llava import analyze_image, encode_image_to_base64, start_ollama_server, stop_ollama_server
# from scripts.metadata_and_signature import extract_metadata, detect_model_signature
# from typing import Optional

# # Charger le modèle EfficientNet-B3
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.efficientnet_b3()
# model.classifier = nn.Sequential(
#     nn.Dropout(0.6),
#     nn.Linear(model.classifier[1].in_features, 256),
#     nn.ReLU(),
#     nn.Linear(256, 1)  # Sortie pour la classification binaire
# )
# model.load_state_dict(torch.load('efficientnetB3_weights_final.pth'))
# model = model.to(device)
# model.eval()

# # Fonction de transformation des images
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisation standard pour les modèles pré-entraînés
# ])

# # Fonction pour afficher la carte de chaleur générée par Grad-CAM sur l'image
# def show_cam_on_image(img, heatmap, output_file='output_cam.png'):
#     img = np.array(img)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam_img = heatmap + np.float32(img) / 255
#     cam_img = cam_img / np.max(cam_img)
#     plt.imshow(cam_img)
#     plt.savefig(output_file)
#     print(f"Carte de chaleur sauvegardée sous {output_file}")

# # Fonction pour effectuer une analyse globale de la carte de chaleur et générer une explication
# def analyze_global_image(heatmap, threshold=0.5):
#     mean_activation = np.mean(heatmap)

#     if mean_activation > threshold:
#         explanation = (
#             "L'image présente des zones d'activation importantes, suggérant des modifications possibles. "
#             "Cela pourrait indiquer que l'image a été retouchée ou altérée."
#         )
#     else:
#         explanation = (
#             "L'image ne montre pas de signes de modifications visibles. "
#             "Elle paraît authentique et non retouchée."
#         )
    
#     return explanation

# # Fonction pour générer un commentaire clair pour les clients
# def generate_global_comment(heatmap, explanation):
#     custom_prompt = (
#         f"Après analyse, il a été constaté que {explanation}. "
#         "Pouvez-vous détailler les raisons pour lesquelles l'image pourrait sembler retouchée ou modifiée ? "
#         "Expliquez pourquoi cela pourrait affecter la qualité ou l'authenticité du produit."
#     )

#     response = ollama.generate(model="llava", prompt=custom_prompt)
#     return response.get('response', "Aucune explication disponible pour le moment.")

# # Fonction pour vérifier si l'image a des incohérences logiques ou est modifiée
# def check_image_modification(image_path):
#     start_ollama_server()

#     image_base64 = encode_image_to_base64(image_path)

#     custom_prompt = (
#         "Analyse cette image et indique 'Oui' si elle contient des éléments incohérents ou des signes de retouche, "
#         "comme des objets illogiques ou une utilisation suspecte de retouches d'image. Réponds par 'Non' si l'image est normale. "
#         "Si tu réponds 'Oui', précise l'incohérence que tu as trouvée."
#     )
#     response = analyze_image(image_base64, custom_prompt).strip()
#     response_lower = response.lower()

#     if response_lower.startswith('oui'):
#         explanation = response[response.find('.')+1:].strip()
#         if not explanation:
#             explanation = "Une incohérence ou modification a été détectée."
#         print(f"Incohérence détectée : {explanation}")
#         return True
#     elif response_lower.startswith('non'):
#         print("Aucune incohérence ou modification détectée.")
#         return False
#     else:
#         print(f"Réponse inattendue de LLaVA : {response}")
#         return False

# # Initialisation de l'application FastAPI
# app = FastAPI()

# @app.post("/analyze_image/")
# async def analyze_image_upload(
#     file: UploadFile = File(...),
#     description: Optional[str] = Form(None)
# ):
#     try:
#         temp_dir = "temp"
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)

#         image_path = f"{temp_dir}/{file.filename}"

#         with open(image_path, "wb") as buffer:
#             buffer.write(await file.read())

#         metadata = extract_metadata(image_path)

#         if metadata:
#             print("Métadonnées détectées :")
#             for key, value in metadata.items():
#                 print(f"{key}: {value}")

#             if detect_model_signature(metadata):
#                 return {"message": "L'image est retouchée ou générée par IA. Analyse interrompue."}
#             else:
#                 if check_image_modification(image_path):
#                     return {"message": "L'image contient des incohérences ou des retouches visibles."}
#                 else:
#                     image = Image.open(image_path)
#                     image_tensor = transform(image).unsqueeze(0).to(device)

#                     outputs = model(image_tensor).squeeze(1)
#                     prediction = torch.sigmoid(outputs).item()

#                     is_truque = prediction > 0.7

#                     if is_truque:
#                         print("L'image semble truquée, application de Grad-CAM...")
#                         grad_cam = GradCAM(model=model, target_layer=model.features[-1])
#                         heatmap = grad_cam.generate_cam(image_tensor)
#                         show_cam_on_image(image, heatmap)

#                         global_analysis = analyze_global_image(heatmap)
#                         global_comment = generate_global_comment(heatmap, global_analysis)

#                         decision = automated_decision(is_truque, global_comment)

#                         # Ajout de la logique pour autoriser le retour de la marchandise
#                         if is_truque:
#                             return_decision = "Le retour de la marchandise est autorisé."
#                         else:
#                             return_decision = "Le retour de la marchandise n'est pas autorisé."

#                         return {
#                             "probability": prediction,
#                             "message": global_comment,
#                             "decision": decision,
#                             "return_decision": return_decision
#                         }
#                     else:
#                         print("L'image semble normale.")

#                         if description:
#                             llava_prompt = (
#                                 f"L'image présente les défauts suivants : {description}. "
#                                 "Ces défauts sont-ils visibles et peuvent-ils être attribués à une mauvaise utilisation ou à des défauts de fabrication ? "
#                                 "Réponds par 'Défauts de fabrication' ou 'Défauts d'utilisation'."
#                             )
#                         else:
#                             llava_prompt = "Analyse l'image et indique s'il y a des défauts visibles et s'ils proviennent d'un défaut de fabrication ou d'une mauvaise utilisation."

#                         llava_response = ollama.generate(model="llava", prompt=llava_prompt)

#                         return {
#                             "probability": prediction,
#                             "message": "L'image semble normale et non modifiée.",
#                             "llava_response": llava_response.get('response')
#                         }

#         else:
#             return {"message": "Impossible d'extraire les métadonnées. Analyse interrompue."}

#     finally:
#         stop_ollama_server()





# import os
# from fastapi import FastAPI, UploadFile, File, Form, Depends
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# from PIL import Image
# import ollama
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from sqlalchemy.orm import Session
# from scripts.database import SessionLocal, ImageAnalysis, Base
# from scripts.grad_cam import GradCAM
# from scripts.rules import automated_decision
# from scripts.llava import analyze_image, encode_image_to_base64, start_ollama_server, stop_ollama_server
# from scripts.metadata_and_signature import extract_metadata, detect_model_signature
# from typing import Optional

# # Initialiser l'application FastAPI
# app = FastAPI()

# # Dépendance pour la session de base de données
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Charger le modèle EfficientNet-B3
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = models.efficientnet_b3()
# model.classifier = nn.Sequential(
#     nn.Dropout(0.6),
#     nn.Linear(model.classifier[1].in_features, 256),
#     nn.ReLU(),
#     nn.Linear(256, 1)
# )
# model.load_state_dict(torch.load('efficientnetB3_weights_final.pth'))
# model = model.to(device)
# model.eval()

# # Transformation des images
# transform = transforms.Compose([
#     transforms.Resize((300, 300)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Fonction pour afficher la carte de chaleur générée par Grad-CAM
# def show_cam_on_image(img, heatmap, output_file='output_cam.png'):
#     img = np.array(img)
#     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
#     heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam_img = heatmap + np.float32(img) / 255
#     cam_img = cam_img / np.max(cam_img)
#     plt.imshow(cam_img)
#     plt.savefig(output_file)
#     print(f"Carte de chaleur sauvegardée sous {output_file}")

# # Analyse globale de la carte de chaleur
# def analyze_global_image(heatmap, threshold=0.5):
#     mean_activation = np.mean(heatmap)
#     if mean_activation > threshold:
#         return "L'image montre des signes de retouche ou altération."
#     return "L'image semble authentique et non retouchée."

# # Générer un commentaire pour les clients
# def generate_global_comment(heatmap, explanation):
#     custom_prompt = (
#         f"Après analyse, il a été constaté que {explanation}. "
#         "Pouvez-vous détailler les raisons pour lesquelles l'image pourrait sembler retouchée ou modifiée ? "
#         "Expliquez pourquoi cela pourrait affecter la qualité ou l'authenticité du produit."
#     )
#     response = ollama.generate(model="llava", prompt=custom_prompt)
#     return response.get('response', "Aucune explication disponible pour le moment.")

# # Vérifier les incohérences de l'image
# def check_image_modification(image_path):
#     start_ollama_server()
#     image_base64 = encode_image_to_base64(image_path)
#     custom_prompt = (
#         "Analyse cette image et indique 'Oui' si elle contient des éléments incohérents ou des signes de retouche, "
#         "comme des objets illogiques ou une utilisation suspecte de retouches d'image. Réponds par 'Non' si l'image est normale."
#     )
#     response = analyze_image(image_base64, custom_prompt).strip().lower()
#     if response.startswith('oui'):
#         return True
#     elif response.startswith('non'):
#         return False
#     else:
#         print(f"Réponse inattendue de LLaVA : {response}")
#         return False

# @app.post("/analyze_image/")
# async def analyze_image_upload(
#     file: UploadFile = File(...),
#     description: Optional[str] = Form(None),
#     db: Session = Depends(get_db)  # Injection de la session de base de données
# ):
#     try:
#         # Enregistrement temporaire de l'image
#         temp_dir = "temp"
#         if not os.path.exists(temp_dir):
#             os.makedirs(temp_dir)
#         image_path = f"{temp_dir}/{file.filename}"
#         with open(image_path, "wb") as buffer:
#             buffer.write(await file.read())

#         # Extraction des métadonnées
#         metadata = extract_metadata(image_path)

#         if metadata and detect_model_signature(metadata):
#             return {"message": "La caractéristique de l'image indique qu'elle a été retouchée ou générée par une IA, ou qu'elle porte une signature de retouche Photoshop."}

#         # Vérification des incohérences logiques
#         if check_image_modification(image_path):
#             return {"message": "L'image contient des incohérences ou des retouches visibles."}

#         # Prédiction avec le modèle
#         image = Image.open(image_path)
#         image_tensor = transform(image).unsqueeze(0).to(device)
#         outputs = model(image_tensor).squeeze(1)
#         prediction = torch.sigmoid(outputs).item()
#         is_truque = prediction > 0.7

#         if is_truque:
#             grad_cam = GradCAM(model=model, target_layer=model.features[-1])
#             heatmap = grad_cam.generate_cam(image_tensor)
#             show_cam_on_image(image, heatmap)
#             global_analysis = analyze_global_image(heatmap)
#             global_comment = generate_global_comment(heatmap, global_analysis)
#             decision = automated_decision(is_truque, global_comment)
#             return_decision = "Le retour de la marchandise n'est pas autorisé." if is_truque else "Le retour de la marchandise n'est pas autorisé."
#         else:
#             global_comment = "L'image semble normale et non modifiée."
#             decision = "Aucune action requise"
#             return_decision = "Le retour de la marchandise est  autorisé."

#         # Sauvegarder les résultats dans la base de données
#         image_analysis = ImageAnalysis(
#             filename=file.filename,
#             probability=prediction,
#             is_truque=is_truque,
#             message=global_comment,
#             decision=decision,
#             return_decision=return_decision
#         )
#         db.add(image_analysis)
#         db.commit()
#         db.refresh(image_analysis)

#         return {
#             "probability": prediction,
#             "message": global_comment,
#             "decision": decision,
#             "return_decision": return_decision,
#             "database_id": image_analysis.id
#         }

#     finally:
#         stop_ollama_server()
#         os.remove(image_path)




import os
from fastapi import FastAPI, UploadFile, File, Form, Depends
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import ollama
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from scripts.database import SessionLocal, ImageAnalysis, Base
from scripts.grad_cam import GradCAM
from scripts.rules import automated_decision
from scripts.llava import analyze_image, encode_image_to_base64, start_ollama_server, stop_ollama_server
from scripts.metadata_and_signature import extract_metadata, detect_model_signature
from typing import Optional

# Initialiser l'application FastAPI
app = FastAPI()

# Dépendance pour la session de base de données
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Charger le modèle EfficientNet-B3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.efficientnet_b3()
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Linear(256, 1)
)
model.load_state_dict(torch.load('efficientnetB3_weights_final.pth'))
model = model.to(device)
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fonction pour afficher la carte de chaleur générée par Grad-CAM
def show_cam_on_image(img, heatmap, output_file='output_cam.png'):
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_img = heatmap + np.float32(img) / 255
    cam_img = cam_img / np.max(cam_img)
    plt.imshow(cam_img)
    plt.savefig(output_file)
    print(f"Carte de chaleur sauvegardée sous {output_file}")

# Analyse globale de la carte de chaleur
def analyze_global_image(heatmap, threshold=0.5):
    mean_activation = np.mean(heatmap)
    if mean_activation > threshold:
        return "L'image montre des signes de retouche ou altération."
    return "L'image semble authentique et non retouchée."

# Générer un commentaire pour les clients
def generate_global_comment(heatmap, explanation):
    custom_prompt = (
        f"Après analyse, il a été constaté que {explanation}. "
        "Pouvez-vous détailler les raisons pour lesquelles l'image pourrait sembler retouchée ou modifiée ? "
        "Expliquez pourquoi cela pourrait affecter la qualité ou l'authenticité du produit."
    )
    response = ollama.generate(model="llava", prompt=custom_prompt)
    return response.get('response', "Aucune explication disponible pour le moment.")

# Vérifier les incohérences de l'image
def check_image_modification(image_path):
    start_ollama_server()
    image_base64 = encode_image_to_base64(image_path)
    custom_prompt = (
        "Analyse cette image et indique 'Oui' si elle contient des éléments incohérents ou des signes de retouche, "
        "comme des objets illogiques ou une utilisation suspecte de retouches d'image. Réponds par 'Non' si l'image est normale."
    )
    response = analyze_image(image_base64, custom_prompt).strip().lower()
    if response.startswith('oui'):
        return True
    elif response.startswith('non'):
        return False
    else:
        print(f"Réponse inattendue de LLaVA : {response}")
        return False

@app.post("/analyze_image/")
async def analyze_image_upload(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db)  # Injection de la session de base de données
):
    try:
        # Enregistrement temporaire de l'image
        temp_dir = "temp"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        image_path = f"{temp_dir}/{file.filename}"
        with open(image_path, "wb") as buffer:
            buffer.write(await file.read())

        # Extraction des métadonnées
        metadata = extract_metadata(image_path)

        if metadata and detect_model_signature(metadata):
            return {"message": "La caractéristique de l'image indique qu'elle a été retouchée ou générée par une IA, ou qu'elle porte une signature de retouche Photoshop."}

        # Vérification des incohérences logiques
        if check_image_modification(image_path):
            return {"message": "L'image contient des incohérences ou des retouches visibles."}

        # Prédiction avec le modèle
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)
        outputs = model(image_tensor).squeeze(1)
        prediction = torch.sigmoid(outputs).item()
        is_truque = prediction > 0.7

        if is_truque:
            grad_cam = GradCAM(model=model, target_layer=model.features[-1])
            heatmap = grad_cam.generate_cam(image_tensor)
            show_cam_on_image(image, heatmap)
            global_analysis = analyze_global_image(heatmap)
            global_comment = generate_global_comment(heatmap, global_analysis)
            decision = automated_decision(is_truque, global_comment)
            return_decision = "Le retour de la marchandise n'est pas autorisé."
        else:
            global_comment = "L'image semble normale et non modifiée."
            decision = "Aucune action requise"
            return_decision = "Le retour de la marchandise est autorisé."

        # Log pour débogage
        print(f"Enregistrement : filename={file.filename}, probability={prediction}, is_truque={is_truque}, "
              f"message={global_comment}, decision={decision}, return_decision={return_decision}")

        # Sauvegarder les résultats dans la base de données
        try:
            image_analysis = ImageAnalysis(
                filename=file.filename,
                probability=prediction,
                is_truque=is_truque,
                message=global_comment,
                decision=decision,
                return_decision=return_decision
            )
            db.add(image_analysis)
            db.commit()
            db.refresh(image_analysis)
        except SQLAlchemyError as e:
            db.rollback()
            print(f"Erreur SQLAlchemy lors de l'enregistrement dans la base de données : {str(e)}")
            return {"message": "Erreur lors de l'enregistrement des données dans la base de données."}

        return {
            "probability": prediction,
            "message": global_comment,
            "decision": decision,
            "return_decision": return_decision,
            "database_id": image_analysis.id
        }

    finally:
        stop_ollama_server()
        os.remove(image_path)
