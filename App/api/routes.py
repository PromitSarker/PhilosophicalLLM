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
    3. Provide 3 questions to ask the user to help them reflect on their context and the quote provided.
    """
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_quotes(vectorstore, user_id)

@router.post("/ask")
async def ask_question(
    question: str = Form(...),
    user_id: Optional[str] = Header(default="default"),
):
    """1. Get relevant quotes based on user's current context.
    2. Just give the quote no explanation or anything. Just the quote 
    3. If there's no quote from knowledge base, then use the LLM to fetch a quote from the internet.
    """
    from main import vectorstore  # Import here to avoid circular imports
    return await chat_service.get_response(question, vectorstore, user_id)

@router.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    values = data.get("values", [])
    feelings = data.get("feelings", [])
    challenges = data.get("challenges", [])
    goals = data.get("goals", [])
    alignment_moment = data.get("alignment_moment", "")
    misalignment_moment = data.get("misalignment_moment", "")
    greater_alignment = data.get("greater_alignment", "")
    reflections = data.get("reflections", "")
    question = data.get("question", "")
    user_id = data.get("user_id", "default")
    system_prompt = data.get("system_prompt", None)

    # Update user context
    context = PersonalContext(
        values=values,
        feelings=feelings,
        challenges=challenges,
        goals=goals,
        alignment_moment=alignment_moment,
        misalignment_moment=misalignment_moment,
        greater_alignment=greater_alignment,
        reflections=reflections
    )
    chat_service.update_context(context)

    # If a question is provided, get a response
    if question:
        from main import vectorstore
        return await chat_service.get_response(question, vectorstore, user_id, system_prompt)
    else:
        return {"message": "Context updated. No question provided."}

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


