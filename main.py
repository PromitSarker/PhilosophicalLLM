import os
from pathlib import Path
from fastapi import FastAPI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from App.api.routes import router
from App.core.config import (
    KNOWLEDGE_BASE_PATH,
    EMBEDDING_MODEL as embedding,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Live Journal API",
    description="A RAG-based API for philosophical discussions",
    version="1.0.0"
)

# === Global Initialization ===
vectorstore = None

# === Helper: Load and Embed Document ===
def load_and_embed():
    """
    Loads and embeds the knowledge base document using FAISS vectorstore.
    Handles file checking, document splitting, and vector embedding.
    """
    global vectorstore
    
    # Ensure knowledge base path exists and is resolved
    knowledge_base_path = Path(KNOWLEDGE_BASE_PATH).resolve()
    if not knowledge_base_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {knowledge_base_path}")
    
    try:
        # Initialize text splitter with configuration
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        
        # Load and split document
        loader = TextLoader(str(knowledge_base_path), encoding="utf-8")
        documents = loader.load()
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No valid chunks found in the knowledge base file.")

        # Create vectorstore from chunks
        vectorstore = FAISS.from_documents(chunks, embedding)
        print("Knowledge base loaded and embedded successfully")
        
    except Exception as e:
        print(f"Error in load_and_embed: {str(e)}")
        raise

# Initialize vectorstore on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the vectorstore when the application starts"""
    try:
        load_and_embed()
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise

# Include the router
app.include_router(router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint providing API status"""
    return {
        "status": "online",
        "message": "Philosophical LLM API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "vectorstore_status": "initialized" if vectorstore else "not initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
