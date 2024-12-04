import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import openai

# Configurez votre clé OpenAI
openai.api_key = "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"

# --- Étape 1 : Chargement des données ---
def load_data(filepath):
    data = pd.read_json(filepath, lines=True)
    return data["description"].tolist()

# --- Étape 2 : Prétraitement et segmentation ---
def preprocess_data(descriptions, chunk_size=512, chunk_overlap=128):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = [splitter.split_text(desc) for desc in descriptions]
    return [chunk for sublist in chunks for chunk in sublist]  # Applatir la liste


# --- Étape 3 : Création d'un index vectoriel ---
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings

def create_vector_database(chunks, embeddings):
    client = chromadb.Client()
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.create_collection("product_descriptions", embedding_function=embedding_function)

    for i, chunk in enumerate(chunks):
        collection.add(
            embeddings=[embeddings[i]],
            documents=[chunk],
            metadatas=[{"id": i}]
        )
    return collection

# --- Étape 4 : Recherche dans la base vectorielle ---
def search_documents(query, collection, model):
    query_embedding = model.encode([query])
    results = collection.query(query_embeddings=query_embedding, n_results=5)
    return results["documents"]

# --- Étape 5 : Interaction avec le LLM ---
def generate_response(documents, question):
    prompt = f"""
    Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations suivantes :
    {documents}
    Question : {question}
    Répondez de manière précise et concise.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]

