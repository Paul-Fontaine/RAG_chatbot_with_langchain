import os
import tempfile
import requests
import bs4
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from urllib.parse import urljoin

def scrape_uqac_links(base_url, max_links=900):
    """
    Scrapes up to max_links relevant links from the UQAC management manual page.
    """
    print("Fetching links from the website...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(base_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    links = []
    for tag in soup.find_all("a", href=True):
        link = tag["href"]
        # Ensure the link is absolute
        full_link = urljoin(base_url, link)
        if (full_link.endswith(".pdf") or full_link.startswith("http")) and len(links) < max_links:
                links.append(full_link)
    print(f"Found {len(links)} links.")
    return links

def sanitize_filename(url):
    """
    Sanitizes a URL to create a valid filename.
    """
    from urllib.parse import urlparse
    parsed_url = urlparse(url)
    filename = parsed_url.netloc + parsed_url.path.replace("/", "_").replace(".", "_")
    return filename[:255]  # Limit filename length

def process_pdf_with_pypdf2(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def download_and_extract_data(links):
    """
    Downloads and extracts content from given links.
    """
    all_texts = []
    for i, link in enumerate(links, start=1):
        print(f"Processing link {i}/{len(links)}: {link}")
        try:
            if link.endswith(".pdf"):
                # Process PDFs
                response = requests.get(link, stream=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                    temp_pdf.write(response.content)
                    temp_pdf.close()
                    content = process_pdf_with_pypdf2(temp_pdf.name)
                    all_texts.append(Document(page_content=content, metadata={"source": link}))
                    os.unlink(temp_pdf.name)
            else:
                # Process HTML pages
                loader = WebBaseLoader(
                    web_paths=(link,),
                    bs_kwargs=dict(
                        parse_only=bs4.SoupStrainer(class_=("entry-header", "entry-content"))
                    ),
                )
                content = loader.load()
                all_texts.extend(content)
        except Exception as e:
            print(f"Failed to process link {link}: {e}")
    return all_texts

def persist_to_chroma(all_texts, embeddings_model, db_path):
    """
    Persists the extracted texts into a Chroma database for local storage and search.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_texts)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=db_path)

    # Persist the database
    vectorstore.persist()
    print(f"Data persisted to Chroma database at {db_path}")

def main():
    base_url = "https://www.uqac.ca/mgestion/"
    db_path = "./chroma_index"
    embeddings_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Scrape links
    print("Scraping links from UQAC management manual...")
    links = scrape_uqac_links(base_url, max_links=900)

    # Download and extract data
    print("Downloading and extracting content...")
    all_texts = download_and_extract_data(links)

    # Persist to Chroma database
    print("Persisting data to Chroma database...")
    persist_to_chroma(all_texts, embeddings_model, db_path)

    print("Process completed successfully.")

if __name__ == "__main__":
    main()