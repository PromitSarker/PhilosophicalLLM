import os
from fastapi import APIRouter, Form, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from typing import Optional
from App.model.schemas import PersonalContext, UserProfile
from App.services.chat_service import ChatService, process_user_input

router = APIRouter()
chat_service = ChatService()

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
    """1. Get relevant quotes based on user's current context.
    2. Just give the quote no explanation or anything. Just the quote 
    3. Provide 3 questions to ask the user to help them reflect on their context.
    """
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_quotes(vectorstore, user_id)

@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    user_id: Optional[str] = Header(default="default")
):
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_response(question, vectorstore, user_id)

@router.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    # Expecting data to have: values, feelings, challenges, reflections
    values = data.get("values", [])
    feelings = data.get("feelings", [])
    challenges = data.get("challenges", [])
    reflections = data.get("reflections", {})

    response = process_user_input(values, feelings, challenges, reflections)
    return response

@router.post("/clear_history")
async def clear_chat_history(
    user_id: Optional[str] = Header(default="default")
):
    chat_service.clear_history(user_id)
    return {"message": f"Chat history cleared for user {user_id}"}

@router.get("/history")
async def get_chat_history(
    user_id: Optional[str] = Header(default="default")
):
    return chat_service.get_history(user_id)

@router.get("/")
def read_root():
    from main import vectorstore, KNOWLEDGE_BASE_PATH
    return {
        "message": "RAG API is running",
        "knowledge_base": str(KNOWLEDGE_BASE_PATH),
        "status": "initialized" if vectorstore else "not initialized"
    }


