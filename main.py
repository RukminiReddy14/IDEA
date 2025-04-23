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
