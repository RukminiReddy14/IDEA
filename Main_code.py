import pdfplumber
import faiss
import numpy as np
import ipywidgets as widgets
from IPython.display import display, clear_output
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain_groq import ChatGroq
from google.colab import files
import os
import yaml

# Initialize the FAISS vector store
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# File paths
index_file = "index.faiss"
chunks_file = "chunks.txt"

# Cryptocurrency-related keywords
CRYPTO_KEYWORDS = {"crypto", "cryptocurrency", "bitcoin", "ethereum", "blockchain", "web3",
                   "decentralized", "mining", "token", "NFT", "stablecoin", "defi", "ledger"}

# Load system prompt from YAML
def load_system_prompt():
    with open("advisor_prompts 1.yml", "r") as file:
        config = yaml.safe_load(file)
    return config["system_prompts"]["advisor_llm"]["description"]

# Fetch system prompt
ADVISOR_SYS_PROMPT = load_system_prompt()
print("Advisor System Prompt Loaded Successfully!")

# Groq API Key
os.environ["GROQ_API_KEY"] = "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"


def validate_pdf_content(pdf_path):
    """Check if the PDF contains cryptocurrency-related keywords and return a decision."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text.lower() + "\n"

        # Checking if the PDF contains at least one keyword from the allowed topics
        if not any(keyword in text for keyword in CRYPTO_KEYWORDS):
            print(f"Skipping {pdf_path}: Content is NOT related to cryptocurrency.")
            return False  # this rejects the file

        return True  # this acccepts the file

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return False  # this rejects the file if it cannot be processed


uploaded_document_names = []

def extract_and_store_text(pdf_paths):
    """Extract and store text from valid cryptocurrency PDFs, while keeping track of filenames."""
    global uploaded_document_names
    all_chunks = []
    all_embeddings = []

    for pdf_path in pdf_paths:
        if not validate_pdf_content(pdf_path):
            continue  # Skipping non-crypto files

        uploaded_document_names.append(pdf_path)  #Storing document name

        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

            # Splitting text into chunks
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_text(text)

            # Generating embeddings
            embeddings = embedding_model.encode(chunks)
            embeddings = np.array(embeddings, dtype=np.float32)

            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)

            print(f"Processed {pdf_path}: Successfully added to FAISS.")

        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")

    if not uploaded_document_names:
        raise ValueError("No valid cryptocurrency-related content found. Please upload relevant PDFs.")

    # Merging embeddings into FAISS
    dimension = all_embeddings[0].shape[1]
    index = faiss.IndexFlatL2(dimension)
    for embeddings in all_embeddings:
        index.add(embeddings)

    # Saving FAISS index
    faiss.write_index(index, index_file)

    # Saving text chunks
    with open(chunks_file, "w") as f:
        for chunk in all_chunks:
            f.write(chunk + "\n")

    print("All cryptocurrency-related PDFs processed successfully.")


pdf_paths = ["/content/Emerging_Tech_Bitcoin_Crypto.pdf"]

