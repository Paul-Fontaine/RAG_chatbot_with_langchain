from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma


llm = OllamaLLM(model="mistral")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    collection_name="langchain",
    embedding_function=embeddings,
    persist_directory="./chroma_index",
    create_collection_if_not_exists=False
)
search = vector_store.similarity_search("francais", k=4)
print()