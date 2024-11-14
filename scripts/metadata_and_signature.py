import subprocess
import json

image_path = 'data/test/ia2.jpg'

# Fonction pour extraire les métadonnées EXIF avec ExifTool
def extract_metadata(image_path):
    exiftool_path = r'"C:\\Program Files\\exiftool-12.97_64\\exiftool.exe"'
    exiftool_command = f'{exiftool_path} -j {image_path}'
    result = subprocess.run(exiftool_command, shell=True, capture_output=True, text=True)

    if result.stdout:
        try:
            metadata = json.loads(result.stdout)[0]
            return metadata
        except json.JSONDecodeError:
            print("Erreur lors de la lecture des métadonnées.")
            return None
    else:
        print("Aucune métadonnée trouvée.")
        return None

# Fonction pour vérifier si une image est retouchée ou générée par IA
def detect_model_signature(metadata):
    if metadata:
        # Détecter si l'image est générée par IA ou retouchée selon les métadonnées disponibles
        if metadata.get('ContainsAiGeneratedContent') == 'Yes' or "Photoshop" in metadata.get('Software', ''):
            return True
    return False

# Bloc principal pour exécuter le script
if __name__ == "__main__":
    # Extraire les métadonnées de l'image
    metadata = extract_metadata(image_path)
    
    if metadata:
        print("Métadonnées extraites :")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        # Vérifier si l'image est générée par IA ou retouchée
        if detect_model_signature(metadata):
            print("L'image est générée par IA ou retouchée.Exif presente une modification ou une source venant d un plateforme d IA")
        else:
            print("L'image semble authentique.")
    else:
        print("Impossible d'extraire les métadonnées.")
