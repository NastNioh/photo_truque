# import subprocess
# import requests
# import base64
# import time
# import json
# import sys

# # note: including the start server code in this script for demo purposes. 
# # You might want to separately start the server so that you're not starting the server every time you make the call. 
# def start_ollama_server():
#     try:
#         subprocess.Popen(["ollama", "run", "llava"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print("Starting Ollama server with LLaVa...")
#         time.sleep(5)  # Wait a bit for the server to start
#     except FileNotFoundError:
#         print("Error: Ollama is not installed or not in the PATH.")
#         sys.exit(1)

# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# def analyze_image(image_base64, custom_prompt):
#     url = "http://localhost:11434/api/generate"

#     payload = {
#         "model": "llava",
#         "prompt": custom_prompt,
#         "images": [image_base64]
#     }

#     response = requests.post(url, json=payload)

#     try:
#         # Split the response text into separate lines
#         response_lines = response.text.strip().split('\n')

#         # Extract and concatenate the 'response' part from each line
#         full_response = ''.join(json.loads(line)['response'] for line in response_lines if 'response' in json.loads(line))

#         return full_response
#     except Exception as e:
#         return f"Error: {e}"

# if __name__ == "__main__":
#     # Path to the image file
#     image_path = "data/defaut/2.jpeg"  # Replace with your image path
    
#     # Direct prompt to check for physical defects in the image
#     custom_prompt = "Y a-t-il des défauts physiques visibles dans cette image ?"

#     # Start the server and process the image
#     start_ollama_server()

#     # Encode the image and analyze it
#     image_base64 = encode_image_to_base64(image_path)
#     result = analyze_image(image_base64, custom_prompt)

#     print("Response:", result)


import subprocess
import requests
import base64
import time
import json
import sys

# Vérifie si le serveur Ollama est déjà en marche
def is_ollama_server_running():
    try:
        # Envoie une requête à l'API pour voir si le serveur est actif
        response = requests.get("http://localhost:11434/api/health")
        if response.status_code == 200:
            print("Le serveur Ollama est déjà en marche.")
            return True
        return False
    except requests.ConnectionError:
        return False

# Démarre le serveur Ollama uniquement s'il n'est pas déjà en marche
def start_ollama_server():
    if is_ollama_server_running():
        print("Le serveur Ollama est déjà en cours d'exécution.")
    else:
        try:
            process = subprocess.Popen(["ollama", "run", "llava"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Démarrage du serveur Ollama avec LLaVA...")
            time.sleep(5)  # Attendre un peu que le serveur démarre
        except FileNotFoundError:
            print("Erreur : Ollama n'est pas installé ou n'est pas dans le PATH.")
            sys.exit(1)

# Arrête le serveur Ollama après la requête
def stop_ollama_server():
    try:
        # Tuer le processus Ollama en l'arrêtant via l'API ou en local
        process = subprocess.Popen(["taskkill", "/F", "/IM", "ollama.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process.communicate()
        print("Serveur Ollama arrêté.")
    except Exception as e:
        print(f"Erreur lors de l'arrêt du serveur Ollama : {e}")

# Fonction pour encoder une image en base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Fonction pour envoyer une requête à Ollama et analyser l'image
def analyze_image(image_base64, custom_prompt):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llava",
        "prompt": custom_prompt,
        "images": [image_base64]
    }

    try:
        response = requests.post(url, json=payload)
        response_lines = response.text.strip().split('\n')
        # Extraire et concaténer la partie 'response' de chaque ligne
        full_response = ''.join(json.loads(line)['response'] for line in response_lines if 'response' in json.loads(line))
        return full_response
    except Exception as e:
        return f"Erreur : {e}"

if __name__ == "__main__":
    # Chemin vers l'image à analyser
    image_path = "data/defaut/2.jpeg"  # Remplace avec le chemin de ton image

    # Prompt personnalisé pour analyser les défauts physiques de l'image
    custom_prompt = "Y a-t-il des défauts physiques visibles dans cette image ?"

    # Démarre le serveur Ollama si nécessaire
    start_ollama_server()

    # Encode l'image en base64
    image_base64 = encode_image_to_base64(image_path)

    # Envoie l'image à Ollama pour analyse
    result = analyze_image(image_base64, custom_prompt)

    # Affiche la réponse
    print("Réponse :", result)

    # Arrête le serveur après utilisation
    stop_ollama_server()

