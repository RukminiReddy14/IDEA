# IDEA
# Intelligent Digital Educational Agent (IDEA)

An AI-powered cryptocurrency advisory system that leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)** to provide accurate, accessible, and educational insights into cryptocurrency concepts.

---

## Project Overview

The cryptocurrency domain is complex, filled with rapidly evolving technologies and terminology that can be difficult for both beginners and experienced users to navigate.  
This project develops an **Intelligent Digital Educational Agent (IDEA)** that:

- Processes and validates cryptocurrency-related documents
- Uses semantic search (FAISS + Sentence Transformers) for query-document matching
- Integrates LLMs (via LangChain) to generate contextually accurate responses
- Provides domain-specific advising and explanations tailored for educational purposes

---

## Key Features

- **Document Processing Pipeline** – Extracts, validates, and embeds cryptocurrency-related PDFs  
- **Vector Search with FAISS** – Retrieves semantically similar chunks for user queries  
- **Hybrid RAG + LLM Architecture** – Enhances accuracy and contextual grounding  
- **Model and Methodology Comparison** – Evaluates Llama 3.3, Llama 4 Scout, Qwen QwQ-32B, and Flan-T5 across Prompt Engineering, RAG, and Fine-Tuning  
- **Educational Focus** – Provides level-appropriate explanations and supports progressive learning  

---

## System Architecture

1. **PDF Upload and Validation** – Filters for cryptocurrency-related documents  
2. **Text Extraction and Embedding** – Creates FAISS index with semantic embeddings (`all-MiniLM-L6-v2`)  
3. **Query Handling** – Retrieves top-k relevant chunks  
4. **LLM Response Generation** – Augments prompts with retrieved context and generates accurate answers  
5. **Evaluation Framework** – Compares models and approaches on performance, speed, and accuracy  

---

## Results and Insights

- **Best Model:** Llama 3.3 (70B) with Prompt Engineering (combined score: 0.451)  
- **Fastest Response:** Llama 3.3 (70B) at approximately 0.85 seconds  
- **Strongest Domain Knowledge:** Qwen QwQ-32B  
- **Key Finding:** Well-crafted prompts often outperform RAG and fine-tuning in cryptocurrency advising  

---

## Future Directions

- Fine-tuning on cryptocurrency-specific datasets  
- Optimizing RAG with larger context windows and improved retrieval mechanisms  
- Expanding to specialized domains such as DeFi and NFTs  
- Incorporating real-time market data  
- Multi-language support and student-level adaptation  
- Interactive learning features (quiz mode and progress tracking)  

---

## Technology Stack

- **LLMs:** Llama 3.3 (70B), Llama 4 Scout, Qwen QwQ-32B, Flan-T5  
- **Frameworks:** LangChain, Groq API  
- **Retrieval:** FAISS (Facebook AI Similarity Search)  
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)  
- **Data Processing:** Python (PDF handling, text chunking, evaluation framework)
