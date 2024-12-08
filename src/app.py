import streamlit as st
import json
from filep2 import load_data, clean_descriptions, segment_descriptions, generate_embeddings, create_vector_database, search_documents, generate_response
import os 

st.set_page_config(page_title="Chatbot en Français", layout="wide")

# -------------------
# Configurations
# -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
DATA_PATH = os.path.join(BASE_DIR, "data", "meta.jsonl")  
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "product_descriptions"
BATCH_SIZE = 50
HISTORY_FILE = "conversation_history.json"

# -------------------
# Préparation des données
# -------------------
@st.cache_resource
def initialize_pipeline():
    # Charger les données
    descriptions = load_data(DATA_PATH)
    cleaned_descriptions = clean_descriptions(descriptions)
    flattened_chunks = segment_descriptions(cleaned_descriptions)
    embeddings = generate_embeddings(flattened_chunks, MODEL_NAME, BATCH_SIZE)
    collection = create_vector_database(flattened_chunks, embeddings, COLLECTION_NAME)
    return collection

collection = initialize_pipeline()

# -------------------
# Gestion de l'historique
# -------------------

def load_history(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_history(file_path, history):
    with open(file_path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)

history = load_history(HISTORY_FILE)

# -------------------
# Interface Streamlit
# -------------------


st.title("🤖 Chatbot Amazon")
st.write("Posez vos questions en français uniquement.")

# Zone d'entrée utilisateur
query = st.text_input("Vous :")

if query:
    # Rechercher des documents pertinents
    documents = search_documents(query, collection, MODEL_NAME)
    
    # Générer une réponse
    response = generate_response(query, documents)
    
    # Afficher la réponse
    st.write(f"🤖 Chatbot : {response}")
    
    # Ajouter à l'historique
    history.append({"query": query, "response": response})
    save_history(HISTORY_FILE, history)

# Afficher l'historique des conversations
st.subheader("📜 Historique des conversations")
for item in history:
    st.write(f"**Vous** : {item['query']}")
    st.write(f"**Chatbot** : {item['response']}")
    st.write("---")
