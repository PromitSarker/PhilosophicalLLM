import os
from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
from App.model.schemas import PersonalContext
from App.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/update_context")
async def update_personal_context(context: PersonalContext):
    return chat_service.update_context(context)

@router.post("/ask")
async def ask_question(question: str = Form(...)):
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_response(question, vectorstore)

@router.get("/")
def read_root():
    from main import vectorstore, KNOWLEDGE_BASE_PATH
    return {
        "message": "RAG API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "status": "initialized" if vectorstore else "not initialized"
    }