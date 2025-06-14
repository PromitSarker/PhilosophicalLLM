import os
from fastapi import APIRouter, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from typing import Optional
from App.model.schemas import PersonalContext, UserProfile
from App.services.chat_service import ChatService

router = APIRouter()
chat_service = ChatService()

@router.post("/set_name")
async def set_user_name(
    profile: UserProfile,
    user_id: Optional[str] = Header(default="default")
):
    """Set user's name for the conversation"""
    return chat_service.set_user_name(profile.name, user_id)

@router.get("/get_name")
async def get_user_name(
    user_id: Optional[str] = Header(default="default")
):
    """Get user's stored name"""
    name = chat_service.get_user_name(user_id)
    if name:
        return {"name": name}
    raise HTTPException(status_code=404, detail="User name not found")


@router.post("/update_context")
async def update_personal_context(
    context: PersonalContext,
    user_id: Optional[str] = Header(default="default")
):
    return chat_service.update_context(context)

@router.get("/quotes")
async def get_quotes(
    user_id: Optional[str] = Header(default="default")
):
    """Get relevant quotes based on user's current context."""
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_quotes(vectorstore, user_id)

@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    user_id: Optional[str] = Header(default="default")
):
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_response(question, vectorstore, user_id)

@router.get("/history")
async def get_chat_history(
    user_id: Optional[str] = Header(default="default")
):
    return chat_service.get_history(user_id)

@router.post("/clear_history")
async def clear_chat_history(
    user_id: Optional[str] = Header(default="default")
):
    chat_service.clear_history(user_id)
    return {"message": f"Chat history cleared for user {user_id}"}

@router.get("/")
def read_root():
    from main import vectorstore, KNOWLEDGE_BASE_PATH
    return {
        "message": "RAG API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "status": "initialized" if vectorstore else "not initialized"
    }


