import requests
import logging
from collections import defaultdict
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from App.model.schemas import PersonalContext
from App.core.config import GROQ_API_URL, HEADERS, LLM_CONFIG

# Set up logging
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.user_context = PersonalContext(
            values=[],
            challenges=[],
            goals=[]
        )
        # Dictionary to store conversation history for each user
        self.conversations = defaultdict(list)
        # Maximum number of messages to keep in history
        self.max_history = 10
        
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

    async def get_response(self, question: str, vectorstore, user_id: str = "default") -> dict:
        """Get response for user question using RAG and LLM with conversation memory"""
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

            # Get conversation history for this user
            conversation = self.conversations[user_id]

            # Prepare messages with history
            messages = [
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
                    - Sometimes ask thoughtful questions that promote self-discovery. 
                    - NOT ALWAYS ask questions, ask only when appropriate.
                    - Highlight patterns between values, challenges, and goals
                    - sometime add some quotes from the knowledge base if relevant. Do not overuse the quoting feature
                    - include the quotes in double quotations so they can be easily identified
                    - Use a friendly and supportive tone
                    - Avoid jargon or overly complex language
                    - Focus on understanding the user's perspective
                    - Use the knowledge context to enrich your responses
                    - Avoid giving direct advice or solutions
                    - try to give pin point straight conversation. DO NOT TALK TOO MUCH WHEN IT CAN BE AVOIDED.
                    - do not repeat the question in your answer.
                    Base your responses on both the knowledge context and personal context provided.
                    Focus on understanding rather than advising."""
                }
            ]

            # Add conversation history
            messages.extend(conversation)

            # Add current context and question
            current_message = {
                "role": "user",
                "content": f"""Personal Context:\n{personal_context}\n
                Knowledge Context:\n{knowledge_context}\n
                Question: {question}
                
                Please provide a thoughtful response that connects the question to my personal values and current situation."""
            }
            messages.append(current_message)

            # Prepare payload with conversation history
            payload = {
                **LLM_CONFIG,
                "messages": messages
            }

            # Make API request
            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.text}")
                raise Exception(f"Groq API error: {response.text}")

            result = response.json()
            assistant_message = result["choices"][0]["message"]["content"]

            # Store the conversation
            self.conversations[user_id].append(current_message)
            self.conversations[user_id].append({
                "role": "assistant",
                "content": assistant_message
            })

            # Keep only the last N messages
            if len(self.conversations[user_id]) > self.max_history:
                self.conversations[user_id] = self.conversations[user_id][-self.max_history:]

            return {"answer": assistant_message}
        
        except Exception as e:
            logger.error(f"ChatService error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def clear_history(self, user_id: str = "default") -> None:
        """Clear conversation history for a specific user"""
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            logger.info(f"Cleared conversation history for user {user_id}")

    def get_history(self, user_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a specific user"""
        return self.conversations.get(user_id, [])