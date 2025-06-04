import os
from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import requests
from pathlib import Path

app = FastAPI()

# Initialize Groq API configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# === Global Initialization ===
KNOWLEDGE_BASE_PATH = Path("knowledge_base.md")
# Using HuggingFace's all-MiniLM-L6-v2 model for embeddings
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
vectorstore = None

# === Helper: Load and Embed Document ===
def load_and_embed():
    global vectorstore
    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {KNOWLEDGE_BASE_PATH}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    loader = TextLoader(str(KNOWLEDGE_BASE_PATH), encoding="utf-8")
    documents = loader.load()
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No valid chunks found in the knowledge base file.")

    vectorstore = FAISS.from_documents(chunks, embedding)
    print("Knowledge base loaded and embedded successfully")

# Initialize vectorstore on startup
@app.on_event("startup")
async def startup_event():
    load_and_embed()

# === Ask a Question Endpoint ===
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vectorstore
    if not vectorstore:
        return JSONResponse(
            status_code=500,
            content={"error": "Knowledge base not initialized."}
        )

    try:
        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a thoughtful companion who balances insight with a tiny bit of philosophical tone for practical understanding. 
                    Your responses should:
                    - Be clear and direct while maintaining depth
                    - Connect tiny philosophical concepts to everyday experiences but be straightforward
                    - Avoid being overly abstract or complex
                    - Use simple language while preserving meaningful insights
                    - Dont give much more information if it's not needed or not asked for
                    - Use simple language while preserving meaningful insights
                    - Focus on understanding the core of the question
                    - Avoid giving direct advice or solutions. But answer directly what has been asked
                    - Encourage reflection and personal insight
                    - Maintain a conversational tone
                    
                    Base your responses only on the context provided. Keep responses concise and relatable."""
                },
                {
                    "role": "user",
                    "content": f"""Context:\n{context}\n\n
                    Question: {question}
                    
                    Please provide a clear, thoughtful response that balances philosophical insight with practical understanding."""
                }
            ],
            "temperature": 0.5,  # Reduced for more focused responses
            "max_tokens": 150,   # Increased slightly for complete thoughts
            "top_p": 1,
            "stream": False
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.text}")

        result = response.json()
        return {"answer": result["choices"][0]["message"]["content"]}
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Query processing failed: {str(e)}"}
        )

# === Health Check ===
@app.get("/")
def read_root():
    return {
        "message": "RAG API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "status": "initialized" if vectorstore else "not initialized"
    }
