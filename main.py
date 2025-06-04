import os
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
import requests
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

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

# === Personal Context Model ===
class PersonalContext(BaseModel):
    values: List[str] = Field(default=[], description="List of personal values")
    challenges: List[str] = Field(default=[], description="List of current challenges")
    mood: Optional[str] = Field(default="neutral", description="Current emotional state")
    goals: Optional[List[str]] = Field(default=[], description="List of personal goals")

    class Config:
        json_schema_extra = {
            "example": {
                "values": ["growth", "authenticity", "compassion"],
                "challenges": ["work-life balance", "self-doubt"],
                "mood": "contemplative",
                "goals": ["self-improvement", "meaningful work"]
            }
        }

# Initial user context
user_context = PersonalContext(
    values=[],
    challenges=[],
    goals=[]
)

# === Update Personal Context Endpoint ===
@app.post("/update_context")
async def update_personal_context(context: PersonalContext):
    try:
        global user_context
        user_context = context
        return {"message": "Personal context updated successfully", "context": context.dict()}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

# === Ask a Question Endpoint ===
@app.post("/ask")
async def ask_question(question: str = Form(...)):
    if not question or question.isspace():
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    
    try:
        global vectorstore, user_context
        if not vectorstore:
            return JSONResponse(
                status_code=500,
                content={"error": "Knowledge base not initialized."}
            )

        relevant_docs = vectorstore.similarity_search(question, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create personal context string
        personal_context = f"""
        Personal Values: {', '.join(user_context.values)}
        Current Challenges: {', '.join(user_context.challenges)}
        Current Mood: {user_context.mood}
        Personal Goals: {', '.join(user_context.goals)}
        """

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": """You are a thoughtful companion who understands personal values and life challenges. 
                    Consider the provided personal context while formulating responses.
                    Your approach should:
                    - Honor and reinforce stated personal values
                    - Acknowledge current challenges without trying to solve them directly
                    - Consider emotional state/mood in your response tone
                    - Connect insights to personal situation
                    - Encourage self-reflection about values and challenges
                    - Help explore the relationship between personal values and current challenges
                    - Maintain empathy while staying precise and concise
                    - Ask thoughtful questions that promote self-discovery
                    - Highlight patterns between values, challenges, and goals
                    
                    Base your responses on both the knowledge context and personal context provided.
                    Focus on understanding rather than advising."""
                },
                {
                    "role": "user",
                    "content": f"""Personal Context:\n{personal_context}\n
                    Knowledge Context:\n{context}\n
                    Question: {question}
                    
                    Please provide a thoughtful response that connects the question to my personal values and current situation."""
                }
            ],
            "temperature": 0.3,
            "max_tokens": 100,
            "top_p": 1,
            "stream": False
        }

        response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Groq API error: {response.text}")

        result = response.json()
        return {"answer": result["choices"][0]["message"]["content"]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Health Check ===
@app.get("/")
def read_root():
    return {
        "message": "RAG API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "status": "initialized" if vectorstore else "not initialized"
    }
