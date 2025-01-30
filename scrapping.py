import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from setup import  Chroma, embeddings

vector_store = Chroma(
    collection_name="langchain",
    embedding_function=embeddings,
    persist_directory="./DB",
)

web_loader = WebBaseLoader(
    web_paths=("https://www.uqac.ca/mgestion/chapitre-2/reglement-sur-la-mission-et-les-valeurs-de-luqac/politique-de-developpement-durable/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("entry-header", "entry-content")
        )
    ),
)
web_pages = web_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
web_splits = text_splitter.split_documents(web_pages)

all_splits = web_splits

# Index chunks
_ = vector_store.add_documents(documents=all_splits)
