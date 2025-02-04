from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Initialisation du modèle Mistral
llm = OllamaLLM(model="mistral")

# Initialisation des embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Chargement de la base Chroma existante
vector_store = Chroma(
    persist_directory="./chroma_index",  # Assurez-vous que ce chemin correspond à votre DB existante
    embedding_function=embeddings,
)
