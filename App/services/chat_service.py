import requests
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List
from App.model.schemas import PersonalContext
from App.core.config import GROQ_API_URL, HEADERS, LLM_CONFIG

class ChatService:
    def __init__(self):
        self.user_context = PersonalContext(
            values=[],
            challenges=[],
            goals=[]
        )
        
    def update_context(self, context: PersonalContext):
        """Update the user's personal context"""
        try:
            self.user_context = context
            return {"message": "Personal context updated successfully", "context": context.dict()}
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    def create_context_string(self) -> str:
        """Create a formatted string of the user's personal context"""
        return f"""
        Personal Values: {', '.join(self.user_context.values)}
        Current Challenges: {', '.join(self.user_context.challenges)}
        Current Mood: {self.user_context.mood}
        Personal Goals: {', '.join(self.user_context.goals)}
        """

    async def get_response(self, question: str, vectorstore) -> dict:
        """Get response for user question using RAG and LLM"""
        if not question or question.isspace():
            raise HTTPException(status_code=422, detail="Question cannot be empty")
        
        try:
            if not vectorstore:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Knowledge base not initialized."}
                )

            # Get relevant documents
            relevant_docs = vectorstore.similarity_search(question, k=3)
            knowledge_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            personal_context = self.create_context_string()

            # Prepare payload
            payload = {
                **LLM_CONFIG,
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
                        - also add some quotes from the knowledge base if relevant
                        - include the quotes in double quotations so they can be easily identified
                        - Use a friendly and supportive tone
                        - Avoid jargon or overly complex language
                        - Focus on understanding the user's perspective
                        - Use the knowledge context to enrich your responses
                        - Avoid giving direct advice or solutions
                        Base your responses on both the knowledge context and personal context provided.
                        Focus on understanding rather than advising."""
                    },
                    {
                        "role": "user",
                        "content": f"""Personal Context:\n{personal_context}\n
                        Knowledge Context:\n{knowledge_context}\n
                        Question: {question}
                        
                        Please provide a thoughtful response that connects the question to my personal values and current situation."""
                    }
                ]
            }

            # Make API request
            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Groq API error: {response.text}")

            result = response.json()
            return {"answer": result["choices"][0]["message"]["content"]}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))