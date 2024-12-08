import os
from filep2 import load_data, clean_descriptions, segment_descriptions, generate_embeddings, create_vector_database, search_documents, generate_response
import pandas as pd
import openai
import os 

# -------------------
# Configurations
# -------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
DATA_PATH = os.path.join(BASE_DIR, "data", "meta.jsonl")  
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 50
COLLECTION_NAME = "product_descriptions"

# -------------------
# Étape 1 : Test du chargement des données
# -------------------
def test_load_data():
    print("\n--- Test : Chargement des données ---")
    if not os.path.exists(DATA_PATH):
        print(f"Erreur : Le fichier '{DATA_PATH}' n'existe pas.")
        return False
    
    descriptions = load_data(DATA_PATH)
    if len(descriptions) == 0:
        print("Erreur : Aucune donnée n'a été chargée. Vérifiez le fichier JSONL.")
        return False
    
    print(f"Succès : {len(descriptions)} descriptions chargées.")
    return descriptions

# -------------------
# Étape 2 : Test du nettoyage des descriptions
# -------------------
def test_clean_descriptions(descriptions):
    print("\n--- Test : Nettoyage des descriptions ---")
    cleaned = clean_descriptions(descriptions)
    if len(cleaned) == 0:
        print("Erreur : Aucune description valide après nettoyage.")
        return False
    
    print(f"Succès : {len(cleaned)} descriptions valides après nettoyage.")
    return cleaned

# -------------------
# Étape 3 : Test de la segmentation des descriptions
# -------------------
def test_segment_descriptions(cleaned_descriptions):
    print("\n--- Test : Segmentation des descriptions ---")
    chunks = segment_descriptions(cleaned_descriptions)
    if len(chunks) == 0:
        print("Erreur : Aucune segmentation générée.")
        return False
    
    print(f"Succès : {len(chunks)} chunks générés.")
    return chunks

# -------------------
# Étape 4 : Test de la génération des embeddings
# -------------------
def test_generate_embeddings(chunks):
    print("\n--- Test : Génération des embeddings ---")
    embeddings = generate_embeddings(chunks, MODEL_NAME, BATCH_SIZE)
    
    # Vérifie si le tableau NumPy est vide
    if embeddings.size == 0:  
        print("Erreur : Aucun embedding généré.")
        return False
    
    print(f"Succès : {embeddings.shape[0]} embeddings générés.")  # .shape[0] donne le nombre d'embeddings
    return embeddings

# -------------------
# Étape 5 : Test de la création de la base vectorielle
# -------------------
def test_create_vector_database(chunks, embeddings):
    print("\n--- Test : Création de la base vectorielle ---")
    collection = create_vector_database(chunks, embeddings, COLLECTION_NAME)
    if collection.count() == 0:
        print("Erreur : La collection vectorielle est vide.")
        return False
    
    print(f"Succès : La base vectorielle contient {collection.count()} documents.")
    return collection

# -------------------
# Étape 6 : Test de la recherche de documents pertinents
# -------------------
def test_search_documents(collection, query):
    print("\n--- Test : Recherche de documents pertinents ---")
    documents = search_documents(query, collection, MODEL_NAME)
    if not documents:
        print("Erreur : Aucun document trouvé pour la requête.")
        return False
    
    print(f"Succès : {len(documents)} documents pertinents récupérés.")
    return documents

# -------------------
# Étape 7 : Test de la génération des réponses
# -------------------
def generate_response(query, documents, temperature=0.7, top_p=1.0):
    if not documents:
        return "Je ne sais pas."
    
    # Nettoyer les documents pour s'assurer qu'ils sont des chaînes
    documents = [" ".join(doc) if isinstance(doc, list) else doc for doc in documents]
    
    prompt = f"""
    Vous êtes un assistant intelligent qui répond aux questions en utilisant uniquement les informations suivantes :
    {" ".join(documents)}
    Question : {query}
    Répondez uniquement en utilisant les informations fournies. 
    Si aucune information pertinente n'est disponible, indiquez clairement : 'Je ne sais pas.'
    """
    
    # Appel à l'API OpenAI avec temperature et top_p
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Vous êtes un assistant intelligent."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,  # Contrôle la créativité
        top_p=top_p  # Contrôle la probabilité cumulative
    )
    return response["choices"][0]["message"]["content"]


# -------------------
# Étape 8 : Test des paramètres `temperature` et `top_p`
# -------------------
def test_parameters(query, documents):
    print("\n--- Test : Impact des paramètres `temperature` et `top_p` ---")
    temperatures = [0.1, 0.5, 0.9]
    top_ps = [0.7, 0.9, 1.0]
    results = []

    for temp in temperatures:
        for p in top_ps:
            response = generate_response(query, documents, temperature=temp, top_p=p)
            results.append({"temperature": temp, "top_p": p, "response": response})
            print(f"Température={temp}, top_p={p} => Réponse : {response}")

    # Sauvegarder les résultats dans un fichier CSV pour analyse
    df = pd.DataFrame(results)
    df.to_csv("parameter_tests.csv", index=False)
    print("\nLes résultats ont été enregistrés dans `parameter_tests.csv`.")
    return results


if __name__ == "__main__":
    # Étape 1 : Charger les données
    descriptions = test_load_data()
    if not descriptions:
        exit()
    
    # Étape 2 : Nettoyer les descriptions
    cleaned_descriptions = test_clean_descriptions(descriptions)
    if not cleaned_descriptions:
        exit()
    
    # Étape 3 : Segmenter les descriptions
    chunks = test_segment_descriptions(cleaned_descriptions)
    if not chunks:
        exit()
    
    # Étape 4 : Générer les embeddings
    embeddings = test_generate_embeddings(chunks)
    if embeddings is False or embeddings.size == 0:  # Vérifie si le tableau est vide ou si l'étape a échoué
        exit()
    
    # Étape 5 : Créer la base vectorielle
    collection = test_create_vector_database(chunks, embeddings)
    if not collection:
        exit()
    
    # Étape 6 : Rechercher des documents pertinents
    query = "Tell me about the OnePlus 6T"
    documents = test_search_documents(collection, query)
    if not documents:
        exit()
    
    # Étape 7 : Générer une réponse
    generate_response(query, documents)
    
    # Étape 8 : Tester les paramètres
    test_parameters(query, documents)
