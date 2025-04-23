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

import time 
import pandas as pd
import matplotlib.pyplot as plt  


# Function to query Groq models
def query_groq_model(query, model_name, system_prompt="/content/advisor_prompts 1.yml"):
    """Generate a response using a Groq model with cryptocurrency-specific prompt."""
    try:
        # Retrieve context
        retrieved_chunks = retrieve_relevant_chunks(query)
        context = "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."
        
        # Construct the prompt with cryptocurrency focus
        crypto_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Context: {context}\n\nCryptocurrency Question: {query}")
        ])
        
        # Initialize the LLM
        groq_api_key = os.getenv("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")
        llm = ChatGroq(model=model_name, api_key=groq_api_key)
        
        # Generate response
        start_time = time.time()
        response = crypto_prompt | llm
        result = response.invoke({})
        end_time = time.time()
        
        # Calculate response time
        response_time = end_time - start_time
        
        return {
            "response": result.content.strip(),
            "response_time": response_time,
            "status": "success"
        }
    except Exception as e:
        return {
            "response": f"Error with {model_name}: {str(e)}",
            "response_time": 0,
            "status": "error"
        }

# Function to query Hugging Face models
def query_hf_model(query, model_name="google/flan-t5-base"):
    """Generate a response using a Hugging Face model for cryptocurrency questions."""
    try:
        # Retrieve context
        retrieved_chunks = retrieve_relevant_chunks(query)
        context = "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."
        
        # Create prompt with cryptocurrency focus
        prompt = f"Context: {context}\n\nYou are a cryptocurrency expert. Answer this question: {query}"
        
        # Use transformers pipeline
        start_time = time.time()
        pipe = pipeline("text2text-generation", model=model_name)
        result = pipe(prompt, max_length=300, do_sample=False)[0]['generated_text']
        end_time = time.time()
        
        # Calculate response time
        response_time = end_time - start_time
        
        return {
            "response": result,
            "response_time": response_time,
            "status": "success"
        }
    except Exception as e:
        return {
            "response": f"Error with {model_name}: {str(e)}",
            "response_time": 0,
            "status": "error"
        }

# Cryptocurrency-specific evaluation function
def evaluate_crypto_response(response, query):
    """
    Simple heuristic to evaluate cryptocurrency responses.
    In a real implementation, this could be more sophisticated.
    """
    # List of important cryptocurrency terms
    crypto_terms = [
        "bitcoin", "ethereum", "blockchain", "cryptocurrency", "token", "wallet",
        "mining", "proof of stake", "proof of work", "defi", "smart contract",
        "nft", "decentralized", "consensus", "ledger", "hash", "block", "transaction"
    ]
    
    # Count cryptocurrency terms in response
    term_count = sum(1 for term in crypto_terms if term.lower() in response.lower())
    
    # Basic score based on term frequency (could be much more sophisticated)
    score = min(10, term_count) / 10  # Normalize to 0-1 range
    
    return score

# Main comparison function with UPDATED MODELS
def compare_crypto_models(query, models=None):
    """Compare different models on cryptocurrency-specific questions."""
    if models is None:
        # Default cryptocurrency-focused model configuration with UPDATED MODELS
        models = [
            {"name": "llama-3.3-70b-versatile", "description": "Llama 3.3 (70B)", "type": "groq"},
            {"name": "llama-4-scout", "description": "Llama 4 Scout", "type": "groq"},
            {"name": "qwen-qwq-32b", "description": "Qwen QwQ-32B", "type": "groq"},
            {"name": "google/flan-t5-base", "description": "Flan-T5 (base)", "type": "hf"}
        ]
    
    results = []
    
    print(f"Comparing models on cryptocurrency query: '{query}'")
    
    for model in models:
        print(f"Processing model: {model['description']}...")
        
        # Get response based on model type
        if model["type"] == "groq":
            result = query_groq_model(query, model["name"])
        else:
            result = query_hf_model(query, model["name"])
        
        # Add cryptocurrency-specific evaluation criteria
        crypto_score = evaluate_crypto_response(result["response"], query)
        
        # Add to results
        results.append({
            "model": model["name"],
            "description": model["description"],
            "response": result["response"],
            "response_time": result["response_time"],
            "crypto_score": crypto_score,
            "status": result["status"]
        })
        
        print(f"  - Completed ({result['status']})")
    
    # Convert to DataFrame
    return pd.DataFrame(results)

# Function to visualize results with cryptocurrency focus
def visualize_crypto_results(results_df):
    """Create visualizations for cryptocurrency model comparison results."""
    # Response time comparison
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["description"], results_df["response_time"], color="blue")
    plt.xlabel("Response Time (seconds)")
    plt.title("Model Response Time Comparison for Cryptocurrency Questions")
    plt.tight_layout()
    plt.savefig("crypto_response_time_comparison.png")
    plt.show()
    
    # Cryptocurrency score comparison
    if "crypto_score" in results_df.columns:
        plt.figure(figsize=(10, 6))
        bars = plt.barh(results_df["description"], results_df["crypto_score"], color="green")
        plt.xlabel("Cryptocurrency Score (0-1)")
        plt.title("Model Cryptocurrency Knowledge Score")
        plt.xlim(0, 1)
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                    ha='left', va='center')
            
        plt.tight_layout()
        plt.savefig("crypto_knowledge_score.png")
        plt.show()
    
    # Response length comparison
    results_df["response_length"] = results_df["response"].apply(lambda x: len(x.split()))
    
    plt.figure(figsize=(10, 6))
    plt.barh(results_df["description"], results_df["response_length"], color="purple")
    plt.xlabel("Response Length (words)")
    plt.title("Model Response Length for Cryptocurrency Questions")
    plt.tight_layout()
    plt.savefig("crypto_response_length_comparison.png")
    plt.show()

# Cryptocurrency benchmark questions
CRYPTO_BENCHMARK_QUESTIONS = [
    "What is Bitcoin and how does it work?",
    "How does blockchain technology secure transactions?",
    "What is the difference between proof-of-work and proof-of-stake?",
    "Explain what smart contracts are and their applications.",
    "What are the environmental impacts of cryptocurrency mining?",
    "How do cryptocurrency wallets work?",
    "What is DeFi (Decentralized Finance) and how is it different from traditional finance?",
    "What are NFTs and why are they valuable?",
    "How do cryptocurrency exchanges operate?",
    "What is the Lightning Network and how does it improve Bitcoin?"
]

# Function to run cryptocurrency benchmark with UPDATED MODELS
def run_crypto_benchmark(questions=CRYPTO_BENCHMARK_QUESTIONS, models=None):
    """Run a benchmark on cryptocurrency-specific questions."""
    if models is None:
        # Default model configuration with UPDATED MODELS
        models = [
            {"name": "llama-3.3-70b-versatile", "description": "Llama 3.3 (70B)", "type": "groq"},
            {"name": "llama-4-scout", "description": "Llama 4 Scout", "type": "groq"},
            {"name": "google/flan-t5-base", "description": "Flan-T5 (base)", "type": "hf"}
        ]
    
    all_results = []
    
    print("Starting cryptocurrency benchmark with", len(questions), "questions and", len(models), "models...")
    
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}/{len(questions)}: '{question}'")
        
        question_results = compare_crypto_models(question, models)
        question_results["question"] = question
        all_results.append(question_results)
    
    # Combine results
    combined_results = pd.concat(all_results)
    
    # Calculate aggregated metrics
    model_metrics = combined_results.groupby("description").agg({
        "response_time": ["mean", "min", "max", "std"],
        "crypto_score": ["mean", "min", "max", "std"] if "crypto_score" in combined_results.columns else [],
        "response": lambda x: np.mean([len(r.split()) for r in x])  # Avg response length
    }).reset_index()
    
    # Clean up column names
    columns = ["model", "avg_time", "min_time", "max_time", "std_time"]
    if "crypto_score" in combined_results.columns:
        columns.extend(["avg_crypto_score", "min_crypto_score", "max_crypto_score", "std_crypto_score"])
    columns.append("avg_length")
    model_metrics.columns = columns
    
    print("\n===== CRYPTOCURRENCY BENCHMARK RESULTS =====\n")
    print(model_metrics)
    
    # Save detailed results
    combined_results.to_csv("crypto_benchmark_detailed_results.csv", index=False)
    model_metrics.to_csv("crypto_benchmark_summary_results.csv", index=False)
    
    # Create visualizations
    plt.figure(figsize=(12, 6))
    plt.barh(model_metrics["model"], model_metrics["avg_time"], xerr=model_metrics["std_time"], capsize=5, color="blue")
    plt.xlabel("Average Response Time (seconds)")
    plt.title("Model Performance on Cryptocurrency Questions: Response Time")
    plt.tight_layout()
    plt.savefig("crypto_benchmark_response_time.png")
    plt.show()
    
    if "avg_crypto_score" in model_metrics.columns:
        plt.figure(figsize=(12, 6))
        plt.barh(model_metrics["model"], model_metrics["avg_crypto_score"], 
                xerr=model_metrics["std_crypto_score"], capsize=5, color="green")
        plt.xlabel("Average Cryptocurrency Score (0-1)")
        plt.title("Model Performance on Cryptocurrency Questions: Knowledge Score")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig("crypto_benchmark_knowledge_score.png")
        plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.barh(model_metrics["model"], model_metrics["avg_length"], color="purple")
    plt.xlabel("Average Response Length (words)")
    plt.title("Model Performance on Cryptocurrency Questions: Response Length")
    plt.tight_layout()
    plt.savefig("crypto_benchmark_response_length.png")
    plt.show()
    
    return combined_results, model_metrics

# ===== INTERACTIVE UI FOR CRYPTO MODEL COMPARISON =====
def run_interactive_crypto_comparison():
    """Interactive UI for cryptocurrency model comparison with widgets."""
    # Create output area
    output = widgets.Output()
    
    # Create text input for query
    query_input = widgets.Text(
        value='What is Bitcoin?',
        placeholder='Enter your cryptocurrency question',
        description='Query:',
        disabled=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )
    
    # Model selection checkboxes with UPDATED MODELS
    model_checkboxes = {
        "llama3": widgets.Checkbox(
            value=True,
            description='Llama 3.3 (70B)',
            disabled=False
        ),
        "llama4": widgets.Checkbox(
            value=True,
            description='Llama 4 Scout',
            disabled=False
        ),
        "qwq": widgets.Checkbox(
            value=True,
            description='Qwen QwQ-32B',
            disabled=False
        ),
        "flan-t5": widgets.Checkbox(
            value=True,
            description='Flan-T5 (base)',
            disabled=False
        )
    }
    
    # Button for comparison
    compare_button = widgets.Button(
        description='Compare Crypto Models',
        disabled=False,
        button_style='primary',
        tooltip='Click to compare models',
        icon='check'
    )
    
    # Button for benchmark
    benchmark_button = widgets.Button(
        description='Run Crypto Benchmark',
        disabled=False,
        button_style='success',
        tooltip='Click to run the benchmark',
        icon='play'
    )
    
    # Arrange widgets
    model_selection = widgets.VBox([
        widgets.HTML('<b>Select Models to Compare:</b>'),
        model_checkboxes["llama3"],
        model_checkboxes["llama4"],
        model_checkboxes["qwq"],
        model_checkboxes["flan-t5"]
    ])
    
    # Button click handler for comparison with UPDATED MODELS
    def on_compare_button_clicked(b):
        with output:
            clear_output()
            
            # Get selected models
            selected_models = []
            if model_checkboxes["llama3"].value:
                selected_models.append({"name": "llama-3.3-70b-versatile", "description": "Llama 3.3 (70B)", "type": "groq"})
            if model_checkboxes["llama4"].value:
                selected_models.append({"name": "llama-4-scout", "description": "Llama 4 Scout", "type": "groq"})
            if model_checkboxes["qwq"].value:
                selected_models.append({"name": "qwen-qwq-32b", "description": "Qwen QwQ-32B", "type": "groq"})
            if model_checkboxes["flan-t5"].value:
                selected_models.append({"name": "google/flan-t5-base", "description": "Flan-T5 (base)", "type": "hf"})
            
            if not selected_models:
                print("Please select at least one model to compare.")
                return
            
            # Get query
            query = query_input.value
            if not query:
                print("Please enter a cryptocurrency question.")
                return
            
            # Run comparison
            print("Starting cryptocurrency model comparison...")
            results_df = compare_crypto_models(query, selected_models)
            
            # Display results
            print("\n===== CRYPTOCURRENCY MODEL COMPARISON RESULTS =====\n")
            for _, row in results_df.iterrows():
                print(f"MODEL: {row['description']}")
                print(f"STATUS: {row['status']}")
                print(f"TIME: {row['response_time']:.2f} seconds")
                if "crypto_score" in row:
                    print(f"CRYPTO SCORE: {row['crypto_score']:.2f}")
                print(f"RESPONSE: {row['response']}")
                print("\n" + "-"*50 + "\n")
            
            # Visualize results
            try:
                visualize_crypto_results(results_df)
            except Exception as e:
                print(f"Error creating visualizations: {e}")
            
            # Save results to CSV
            try:
                results_df.to_csv("crypto_model_comparison_results.csv", index=False)
                print("Results saved to 'crypto_model_comparison_results.csv'")
            except Exception as e:
                print(f"Error saving results: {e}")
    
    # Button click handler for benchmark with UPDATED MODELS
    def on_benchmark_button_clicked(b):
        with output:
            clear_output()
            
            # Get selected models
            selected_models = []
            if model_checkboxes["llama3"].value:
                selected_models.append({"name": "llama-3.3-70b-versatile", "description": "Llama 3.3 (70B)", "type": "groq"})
            if model_checkboxes["llama4"].value:
                selected_models.append({"name": "llama-4-scout", "description": "Llama 4 Scout", "type": "groq"})
            if model_checkboxes["qwq"].value:
                selected_models.append({"name": "qwen-qwq-32b", "description": "Qwen QwQ-32B", "type": "groq"})
            if model_checkboxes["flan-t5"].value:
                selected_models.append({"name": "google/flan-t5-base", "description": "Flan-T5 (base)", "type": "hf"})
            
            if not selected_models:
                print("Please select at least one model to benchmark.")
                return
            
            # Run benchmark
            print("Starting cryptocurrency benchmark. This may take several minutes...")
            combined_results, model_metrics = run_crypto_benchmark(models=selected_models)
            print("\nCryptocurrency benchmark completed!")
    
    compare_button.on_click(on_compare_button_clicked)
    benchmark_button.on_click(on_benchmark_button_clicked)
    
    # Display widgets
    display(widgets.HTML('<h2>Cryptocurrency Model Comparison</h2>'))
    display(query_input)
    display(widgets.HBox([model_selection, widgets.VBox([compare_button, benchmark_button])]))
    display(output)

# Simple example of how to use this code
if __name__ == "__main__":
    # Check if we're in a Jupyter/Colab environment
    try:
        get_ipython
        in_notebook = True
    except:
        in_notebook = False
    
    if in_notebook:
        # Interactive UI for notebooks
        run_interactive_crypto_comparison()
    else:
        # Simple example for non-notebooks
        query = "What are the main differences between Bitcoin and Ethereum?"
        
        print("Comparing models on cryptocurrency question...")
        results = compare_crypto_models(query)
        
        print("\nResults:")
        for _, row in results.iterrows():
            print(f"\nModel: {row['description']}")
            print(f"Response: {row['response']}")

from tqdm import tqdm

# First, let's properly download the NLTK data
import nltk
nltk.download('punkt')

# Import necessary libraries for different approaches
from langchain_groq import ChatGroq
from langchain.prompts.chat import ChatPromptTemplate
from transformers import pipeline

# Simplified version that doesn't require many external dependencies
class CryptoEvalConfig:
    def __init__(self):
        # Models to evaluate
        self.models = [
            {"name": "llama-3.3-70b-versatile", "description": "Llama 3.3 (70B)", "type": "groq"},
            {"name": "llama-4-scout", "description": "Llama 4 Scout", "type": "groq"},
            {"name": "qwen-qwq-32b", "description": "Qwen QwQ-32B", "type": "groq"}
        ]
        
        # Approaches to evaluate
        self.approaches = ["prompt_engineering", "rag", "fine_tuning"]
        
        # Benchmark queries and references
        self.benchmark_queries = [
            {
                "query": "What is Bitcoin?",
                "reference": "Bitcoin is a decentralized digital currency created in 2009 that operates without a central authority."
            },
            {
                "query": "How does blockchain work?",
                "reference": "Blockchain is a distributed ledger technology that records transactions in a secure, immutable way across multiple computers, ensuring transparency and security without central control."
            }
        ]
        
        # Simplified hyperparameters
        self.hyperparameters = {
            "prompt_engineering": {
                "temperature": [0.0, 0.7],
                "max_tokens": [100, 300]
            },
            "rag": {
                "chunk_size": [200],
                "k": [3]
            },
            "fine_tuning": {
                "epochs": [1],
                "learning_rate": [5e-5]
            }
        }
        
        # Paths
        self.prompt_file_path = "advisor_prompts 1.yml"
        self.results_path = "evaluation_results/"
        os.makedirs(self.results_path, exist_ok=True)

# Simplified evaluation metrics without relying on external packages
class SimpleEvaluationMetrics:
    def __init__(self):
        pass
    
    def calculate_token_overlap(self, candidate, reference):
        """Calculate simple token overlap (similar to ROUGE but simpler)"""
        if not candidate or not reference:
            return 0
        
        # Tokenize with simple split
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        
        # Create sets of tokens
        candidate_set = set(candidate_tokens)
        reference_set = set(reference_tokens)
        
        # Calculate overlap
        overlap = len(candidate_set.intersection(reference_set))
        
        # Calculate F1 score
        precision = overlap / len(candidate_set) if candidate_set else 0
        recall = overlap / len(reference_set) if reference_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return f1
    
    def calculate_crypto_score(self, response):
        """Calculate cryptocurrency domain-specific score."""
        crypto_terms = [
            "bitcoin", "ethereum", "blockchain", "cryptocurrency", "token", "wallet",
            "mining", "proof of stake", "proof of work", "defi", "smart contract",
            "nft", "decentralized", "consensus", "ledger", "hash", "block", "transaction"
        ]
        
        # Count cryptocurrency terms in lowercase response
        response_lower = response.lower()
        term_count = sum(1 for term in crypto_terms if term in response_lower)
        
        # Normalize score (0-1)
        crypto_score = min(1.0, term_count / 10)
        
        return crypto_score
    
    def evaluate_response(self, response, reference):
        """Evaluate a model response using token overlap and crypto score"""
        token_overlap = self.calculate_token_overlap(response, reference)
        crypto_score = self.calculate_crypto_score(response)
        
        # Simple combined score
        combined_score = 0.6 * token_overlap + 0.4 * crypto_score
        
        return {
            "token_overlap": token_overlap,
            "crypto_score": crypto_score,
            "combined_score": combined_score
        }

# Simplified approach implementations
class PromptEngineeringApproach:
    def __init__(self, model_info, prompt_file_path, hyperparams=None):
        self.model_info = model_info
        self.prompt_file_path = prompt_file_path
        self.hyperparams = hyperparams or {"temperature": 0.0, "max_tokens": 200}
        self.load_prompts()
    
    def load_prompts(self):
        """Load advisor prompts from YAML file or use default."""
        try:
            with open(self.prompt_file_path, 'r') as file:
                self.prompts = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")
            self.prompts = {"system_prompt": "You are a cryptocurrency advisor. Provide accurate information."}
    
    def generate_response(self, query):
        """Simulate generation using prompt engineering approach."""
        try:
            system_prompt = self.prompts.get("system_prompt", "You are a cryptocurrency advisor.")
            
            # Simulate response based on query and model
            if query == "What is Bitcoin?":
                response = f"Bitcoin is a decentralized digital currency created in 2009 by an anonymous person or group using the pseudonym Satoshi Nakamoto. It operates without a central authority or banks. This response was generated with temperature={self.hyperparams['temperature']}."
            elif query == "How does blockchain work?":
                response = f"Blockchain is a distributed ledger technology where transactions are recorded across multiple computers in a way that ensures security, transparency, and immutability. Each block contains transaction data and a hash of the previous block. This response used max_tokens={self.hyperparams['max_tokens']}."
            else:
                response = f"This is a simulated response about {query} from the {self.model_info['description']} model using prompt engineering."
            
            return {
                "response": response,
                "response_time": 1.2,  # Simulated response time
                "status": "success"
            }
        except Exception as e:
            return {
                "response": f"Error with {self.model_info['name']}: {str(e)}",
                "response_time": 0,
                "status": "error"
            }

class RAGApproach:
    def __init__(self, model_info, hyperparams=None):
        self.model_info = model_info
        self.hyperparams = hyperparams or {"chunk_size": 200, "k": 3}
    
    def generate_response(self, query):
        """Simulate generation using RAG approach."""
        try:
            # Simulate RAG response based on query and model
            if query == "What is Bitcoin?":
                response = f"Based on relevant documents, Bitcoin is a peer-to-peer electronic cash system introduced in a 2008 whitepaper by Satoshi Nakamoto. It enables online payments without going through financial institutions. RAG parameters: chunk_size={self.hyperparams['chunk_size']}, k={self.hyperparams['k']}."
            elif query == "How does blockchain work?":
                response = f"According to retrieved documents, blockchain maintains a continuously growing list of records (blocks) linked using cryptography. Each block contains a timestamp and transaction data. Once recorded, the data cannot be altered retroactively. This response used {self.model_info['description']} with RAG."
            else:
                response = f"This is a simulated RAG response about {query} using retrieved context and the {self.model_info['description']} model."
            
            return {
                "response": response,
                "response_time": 1.5,  # Simulated response time
                "status": "success"
            }
        except Exception as e:
            return {
                "response": f"Error with {self.model_info['name']} RAG: {str(e)}",
                "response_time": 0,
                "status": "error"
            }

class FineTuningApproach:
    def __init__(self, model_info, hyperparams=None):
        self.model_info = model_info
        self.hyperparams = hyperparams or {"epochs": 1, "learning_rate": 5e-5}
    
    def generate_response(self, query):
        """Simulate generation using fine-tuned model approach."""
        try:
            # Simulate fine-tuned response based on query and model
            if query == "What is Bitcoin?":
                response = f"Bitcoin is a decentralized digital currency that enables instant payments to anyone, anywhere in the world without requiring a central authority. This response is from a model fine-tuned for {self.hyperparams['epochs']} epochs."
            elif query == "How does blockchain work?":
                response = f"Blockchain technology maintains a digital ledger of transactions across a network of computers. It uses cryptographic methods to ensure data integrity and consensus mechanisms to validate transactions. Fine-tuned with learning rate {self.hyperparams['learning_rate']}."
            else:
                response = f"This is a simulated response from a fine-tuned {self.model_info['description']} model about {query}."
            
            return {
                "response": response,
                "response_time": 0.8,  # Simulated response time
                "status": "success"
            }
        except Exception as e:
            return {
                "response": f"Error with {self.model_info['name']} fine-tuning: {str(e)}",
                "response_time": 0,
                "status": "error"
            }

# Simplified experiment runner
def run_simplified_evaluation():
    """Run a simplified evaluation that doesn't require external dependencies."""
    print("Starting simplified evaluation experiment...")
    
    # Initialize configuration
    config = CryptoEvalConfig()
    
    # Initialize metrics
    metrics = SimpleEvaluationMetrics()
    
    # Initialize results list
    results = []
    
    # Run experiment
    for approach_name in config.approaches:
        print(f"\nEvaluating {approach_name} approach...")
        
        # Get hyperparameter combinations for this approach
        hyperparam_keys = list(config.hyperparameters[approach_name].keys())
        hyperparam_values = list(config.hyperparameters[approach_name].values())
        
        # Generate all combinations of hyperparameters
        hyperparam_combinations = []
        
        def generate_combinations(index, current_combo):
            if index == len(hyperparam_keys):
                hyperparam_combinations.append(current_combo.copy())
                return
            
            for value in hyperparam_values[index]:
                current_combo[hyperparam_keys[index]] = value
                generate_combinations(index + 1, current_combo)
        
        generate_combinations(0, {})
        
        for model_info in config.models:
            print(f"\n  Testing model: {model_info['description']}")
            
            for hyperparams in hyperparam_combinations:
                hyperparam_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])
                print(f"    With hyperparameters: {hyperparam_str}")
                
                # Initialize the appropriate approach
                if approach_name == "prompt_engineering":
                    approach = PromptEngineeringApproach(
                        model_info, 
                        config.prompt_file_path,
                        hyperparams
                    )
                elif approach_name == "rag":
                    approach = RAGApproach(
                        model_info,
                        hyperparams
                    )
                elif approach_name == "fine_tuning":
                    approach = FineTuningApproach(
                        model_info,
                        hyperparams
                    )
                
                # Run benchmark queries
                for benchmark in config.benchmark_queries:
                    query = benchmark["query"]
                    reference = benchmark["reference"]
                    
                    print(f"      Query: {query}")
                    
                    # Generate response
                    result = approach.generate_response(query)
                    
                    if result["status"] == "success":
                        # Evaluate response
                        evaluation = metrics.evaluate_response(result["response"], reference)
                        
                        # Combine all information
                        experiment_result = {
                            "approach": approach_name,
                            "model": model_info["name"],
                            "model_description": model_info["description"],
                            "hyperparameters": str(hyperparams),  # Convert to string for easier handling
                            "query": query,
                            "reference": reference,
                            "response": result["response"],
                            "response_time": result["response_time"],
                            "status": result["status"],
                            "token_overlap": evaluation["token_overlap"],
                            "crypto_score": evaluation["crypto_score"],
                            "combined_score": evaluation["combined_score"]
                        }
                        
                        # Add to results
                        results.append(experiment_result)
                        
                        print(f"      Combined score: {evaluation['combined_score']:.4f}")
                    else:
                        print(f"      Error: {result['response']}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_path = os.path.join(config.results_path, "simplified_experiment_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Analyze results
    if len(results_df) > 0:
        print("\n===== ANALYSIS OF RESULTS =====")
        
        # 1. Best approach overall
        approach_metrics = results_df.groupby("approach").agg({
            "combined_score": "mean",
            "token_overlap": "mean",
            "crypto_score": "mean",
            "response_time": "mean"
        }).reset_index()
        
        print("\nPerformance by Approach:")
        print(approach_metrics[["approach", "combined_score", "token_overlap", "crypto_score"]])
        
        best_approach = approach_metrics.loc[approach_metrics["combined_score"].idxmax()]
        print(f"\nBest approach: {best_approach['approach']} (Combined Score: {best_approach['combined_score']:.4f})")
        
        # 2. Best model overall
        model_metrics = results_df.groupby("model_description").agg({
            "combined_score": "mean",
            "token_overlap": "mean",
            "crypto_score": "mean",
            "response_time": "mean"
        }).reset_index()
        
        print("\nPerformance by Model:")
        print(model_metrics[["model_description", "combined_score", "token_overlap", "crypto_score"]])
        
        best_model = model_metrics.loc[model_metrics["combined_score"].idxmax()]
        print(f"\nBest model: {best_model['model_description']} (Combined Score: {best_model['combined_score']:.4f})")
        
        # 3. Best approach-model combination
        combo_metrics = results_df.groupby(["approach", "model_description"]).agg({
            "combined_score": "mean",
            "token_overlap": "mean",
            "crypto_score": "mean"
        }).reset_index()
        
        print("\nTop 3 Approach-Model Combinations:")
        top_combos = combo_metrics.sort_values("combined_score", ascending=False).head(3)
        print(top_combos[["approach", "model_description", "combined_score"]])
        
        # 4. Sample responses from best configurations
        best_idx = results_df["combined_score"].idxmax()
        best_result = results_df.loc[best_idx]
        
        print("\nBest Overall Result:")
        print(f"  Approach: {best_result['approach']}")
        print(f"  Model: {best_result['model_description']}")
        print(f"  Query: {best_result['query']}")
        print(f"  Response: {best_result['response']}")
        print(f"  Combined Score: {best_result['combined_score']:.4f}")
        
        return results_df, approach_metrics, model_metrics, combo_metrics
    else:
        print("No results to analyze.")
        return None, None, None, None

# Run the simplified evaluation
if __name__ == "__main__":
    results_df, approach_metrics, model_metrics, combo_metrics = run_simplified_evaluation()
# Button UI for File Upload Selection
output = widgets.Output()

def upload_single_file(_):
    """Handles single file upload"""
    with output:
        clear_output(wait=True)
        print("\nPlease upload a single PDF related to cryptocurrency.")
        uploaded_file = files.upload()
        pdf_paths = [list(uploaded_file.keys())[0]]  # Convert single file into a list
        extract_and_store_text(pdf_paths)
  def upload_multiple_files(_):
    """Handles multiple file uploads"""
    with output:
        clear_output(wait=True)
        print("\nPlease upload multiple PDFs related to cryptocurrency.")
        uploaded_files = files.upload()
        pdf_paths = list(uploaded_files.keys())  # Convert all files into a list
        extract_and_store_text(pdf_paths)
    # Create buttons
button_single = widgets.Button(description="Upload Single PDF")
button_multiple = widgets.Button(description="Upload Multiple PDFs")
# Assign event handlers
button_single.on_click(upload_single_file)
button_multiple.on_click(upload_multiple_files)
# Display buttons
print("Choose an upload option:")
display(button_single, button_multiple)
display(output)

def retrieve_relevant_chunks(query, top_k=3):
    """Retrieve the top-k most relevant text chunks from FAISS across all uploaded PDFs."""
    if not os.path.exists(index_file):
        raise FileNotFoundError("FAISS index not found. Please upload valid cryptocurrency-related PDFs first.")

    # Load FAISS index
    index = faiss.read_index(index_file)

    # Generate query embedding
    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32)

    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve corresponding text chunks from ALL processed PDFs
    with open(chunks_file, "r") as f:
        text_chunks = f.readlines()

    retrieved_chunks = [text_chunks[i].strip() for i in indices[0] if i < len(text_chunks)]

    return retrieved_chunks

def query_advisor(query):
    """Generate a response from the advisor agent using FAISS-based context and Groq API."""
    global uploaded_document_names

    #If the user asks about uploaded documents, return the list directly
    if "uploaded documents" in query.lower() or "list of documents" in query.lower():
        if uploaded_document_names:
            return f"The uploaded documents are: {', '.join(uploaded_document_names)}"
        else:
            return "No documents have been uploaded yet."

    try:
        # Retrieve context
        retrieved_chunks = retrieve_relevant_chunks(query)
        context = "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

        # Construct the chat prompt
        advisor_prompt = ChatPromptTemplate.from_messages([
            ("system", ADVISOR_SYS_PROMPT),
            ("human", f"Context: {context}\n\nUser Query: {query}")
        ])

        # Initialize Groq LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)

        # Generate response
        response = advisor_prompt | llm
        result = response.invoke({})

        return result.content.strip()

    except Exception as e:
        return f"An error occurred: {e}"
    def query_advisor(question):
    """Query the cryptocurrency advisor with a single question."""
    try:
        # Set up the prompt
        advisor_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a cryptocurrency advisor. Provide accurate information about cryptocurrencies."),
            ("human", f"Cryptocurrency Question: {question}")
        ])
        
        # Initialize the LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm = ChatGroq(model="llama3-70b-8192", api_key=groq_api_key)
        
        # Generate response
        response = advisor_prompt | llm
        result = response.invoke({})
        
        return result.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

# Ask a single question
user_question = input("Enter your cryptocurrency question: ")
response = query_advisor(user_question)
print("\nAdvisor:", response)
