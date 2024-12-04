import streamlit as st
from src.filep2 import load_data, preprocess_data, create_embeddings, create_vector_database, search_documents, generate_response
from sentence_transformers import SentenceTransformer

# --- Configuration Streamlit ---
st.title("Product Description Assistant (RAG)")
st.write("Posez des questions sur les descriptions de produits !")

# --- Charger les données ---
data_path = "data/meta.jsonl"
st.write("Chargement des données...")
descriptions = load_data(data_path)

# --- Prétraitement et segmentation ---
st.write("Prétraitement des données...")
chunks = preprocess_data(descriptions)

# --- Création des embeddings et de la base vectorielle ---
st.write("Création des embeddings et de la base vectorielle...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = create_embeddings(chunks)
collection = create_vector_database(chunks, embeddings)

# --- Interface utilisateur ---
question = st.text_input("Entrez votre question :")

if question:
    st.write("Recherche des documents pertinents...")
    documents = search_documents(question, collection, model)
    
    st.write("Génération de la réponse...")
    response = generate_response(documents, question)
    
    st.subheader("Réponse :")
    st.write(response)
    
    st.subheader("Documents utilisés :")
    for doc in documents:
        st.write(f"- {doc}")
