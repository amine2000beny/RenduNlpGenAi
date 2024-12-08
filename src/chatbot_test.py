from filep2 import load_data, clean_descriptions, segment_descriptions, generate_embeddings, create_vector_database, search_documents, generate_response
import os
# -------------------
# Configurations
# -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
DATA_PATH = os.path.join(BASE_DIR, "data", "meta.jsonl")  
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "product_descriptions"
BATCH_SIZE = 50

# -------------------
# Charger et préparer les données
# -------------------
print("Chargement des données...")
descriptions = load_data(DATA_PATH)

print("Nettoyage des descriptions...")
cleaned_descriptions = clean_descriptions(descriptions)

print("Segmentation des descriptions...")
flattened_chunks = segment_descriptions(cleaned_descriptions)

print("Génération des embeddings...")
embeddings = generate_embeddings(flattened_chunks, MODEL_NAME, BATCH_SIZE)

print("Création de la base vectorielle...")
collection = create_vector_database(flattened_chunks, embeddings, COLLECTION_NAME)

# -------------------
# Interface chatbot
# -------------------
print("\nChatbot prêt. Posez vos questions (tapez 'exit' pour quitter) !\n")

while True:
    query = input("Vous : ")
    if query.lower() == "exit":
        print("Merci d'avoir utilisé le chatbot. À bientôt !")
        break

    # Rechercher des documents pertinents
    documents = search_documents(query, collection, MODEL_NAME)
    
    # Générer une réponse
    response = generate_response(query, documents)
    
    print(f"Chatbot : {response}")
