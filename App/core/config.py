import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

# File paths
KNOWLEDGE_BASE_PATH = Path("knowledge_base.md")

# Model Configuration
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# Model Parameters
LLM_CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.5,
    "max_tokens": 500,
    "top_p": 1,
    "stream": False
}

# Text Splitting Parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200