from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


llm = OllamaLLM(model="mistral")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="manuel_de_gestion",
    embedding_function=embeddings,
    persist_directory="./DB"
)

print("setup done")
