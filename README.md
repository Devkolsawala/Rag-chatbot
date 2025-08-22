# 🤖 RAG Chatbot - Company Knowledge Base

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit that helps employees find information about company policies and procedures.

## 🌟 Features

- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **Fast Vector Search**: FAISS-powered similarity search
- **Interactive UI**: Clean Streamlit interface
- **Conversation Memory**: Maintains chat history during session
- **Smart Caching**: Embeddings cached for faster startup
- **Error Recovery**: Robust error handling and cache management

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Embeddings**: all-MiniLM-L6-v2 (SentenceTransformers)
- **Vector Database**: FAISS (CPU optimized)
- **Backend**: Python 3.8+

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Devkolsawala/rag-chatbot.git
cd rag-chatbot
