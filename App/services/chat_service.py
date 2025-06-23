import requests
import logging
from collections import defaultdict
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict
from App.model.schemas import PersonalContext
from App.core.config import GROQ_API_URL, HEADERS, LLM_CONFIG
import random  # Add this with other imports

# Set up logging with basic configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        # Initialize with default PersonalContext
        self.user_context = PersonalContext(
            values=[],
            challenges=[],
            goals=[]
        )
        # Use defaultdict for automatic initialization of new user conversations
        self.conversations = defaultdict(list)
        self.user_names = {}
        self.max_history = 10  # Maximum number of messages to keep in history

    def set_user_name(self, name: str, user_id: str = "default") -> dict:
        """Store user's name"""
        self.user_names[user_id] = name
        return {"message": f"Hello, {name}! I'll remember you."}
    
    def get_user_name(self, user_id: str = "default") -> str:
        """Get user's name if stored"""
        return self.user_names.get(user_id, None)

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

            # Get the user's name
            user_name = self.user_names.get(user_id, "")

            # Get relevant documents
            relevant_docs = vectorstore.similarity_search(question, k=3)
            knowledge_context = "\n\n".join([doc.page_content for doc in relevant_docs])
            personal_context = self.create_context_string()

            # Initialize conversation history if needed
            if user_id not in self.conversations:
                self.conversations[user_id] = []

            # Get conversation history for this user
            conversation = self.conversations[user_id]

            # Add user name to the system message
            system_message = {
                "role": "system",
                "content": f"""You are a thoughtful companion who understands personal values and life challenges. 
                The user's name is {user_name}. Address them by name when appropriate.
                Consider the provided personal context while formulating responses.
                Your approach should:
                - Do not repeat the question. Sometimes do change the greeting part
                - Honor and reinforce stated personal values
                - Acknowledge current challenges without trying to solve them directly
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

            # Prepare messages with history
            messages = [system_message]
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
        """Clear conversation history and user data"""
        if user_id in self.conversations:
            self.conversations[user_id] = []  # Initialize as empty list instead of clear()
        if user_id in self.user_names:
            del self.user_names[user_id]
        logger.info(f"Cleared conversation history for user {user_id}")

    def get_history(self, user_id: str = "default") -> List[Dict[str, str]]:
        """Get conversation history for a specific user"""
        return self.conversations.get(user_id, [])  # Return empty list if user_id not found

    async def get_quotes(self, vectorstore, user_id: str = "default") -> dict:
        """Get relevant quotes based on user's current emotional context.
        - Provide just the quote. No additional information. No nothing, just the quote.
        - Provide just one quote 
        - If there are no relevant quotes, provide a general quote that is relevant to the user's current emotional state and values. But make it consize as well.
        - 

        """
        try:
            if not vectorstore:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Knowledge base not initialized."}
                )

            # Get personal context with focus on emotional state
            context_string = f"""
            Find philosophical quotes relevant to someone who is:
            - Dealing with: {', '.join(self.user_context.challenges)}
            - Working towards: {', '.join(self.user_context.goals)}
            - Values: {', '.join(self.user_context.values)}
            """
            
            # Search for multiple relevant documents to increase variety
            relevant_docs = vectorstore.similarity_search(context_string, k=5)
            
            # Extract quotes from all documents
            quotes = []
            for doc in relevant_docs:
                content = doc.page_content
                start_idx = 0
                
                # Find all quotes in the document
                while True:
                    quote_start = content.find('"', start_idx)
                    if quote_start == -1:
                        break
                        
                    quote_end = content.find('"', quote_start + 1)
                    if quote_end == -1:
                        break
                        
                    quote = content[quote_start + 1:quote_end].strip()
                    if len(quote) > 20:  # Only add quotes of reasonable length
                        quotes.append(quote)
                    start_idx = quote_end + 1

            # If no quotes found with quotation marks, take short passages
            if not quotes:
                quotes = [doc.page_content for doc in relevant_docs]

            # Select a random quote from the collection
            selected_quote = random.choice(quotes) if quotes else "No relevant quote found."

            # Generate AI-powered questions based on the quote and context
            questions = await self._generate_ai_questions(selected_quote)

            return {
                "quote": selected_quote,
                "question1": [questions[0]] if len(questions) > 0 else ["What does this quote mean to you?"],
                "question2": [questions[1]] if len(questions) > 1 else ["How does this relate to your current situation?"],
                "question3": [questions[2]] if len(questions) > 2 else ["What insight can you draw from this?"]
            }

        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving quote: {str(e)}"
            )

    async def _generate_ai_questions(self, quote: str) -> List[str]:
        """Use AI to generate 3 thoughtful questions based on the quote and user's personal context."""
        try:
            personal_context = self.create_context_string()
            
            system_message = {
                "role": "system",
                "content": """You are a thoughtful companion who creates reflective questions. 
                Generate exactly 3 deep, personalized reflection questions based on the provided quote and user's personal context.
                
                Guidelines for questions:
                - Make the question short and to the point precise.
                - Always avoid asking long questions that are too long or complex. 
                - Make them philosophical if you can.
                - Make them specific to the user's values, challenges, and goals
                - Avoid generic questions - make them meaningful and contextual
                - Each question should explore a different aspect (values, challenges, goals, or insights)
                - Keep questions clear and concise
                - Make them thought-provoking but not overwhelming
                
                Return ONLY the 3 questions, each on a separate line, without numbering or bullet points."""
            }
            
            user_message = {
                "role": "user",
                "content": f"""Quote: "{quote}"
                
                User's Personal Context:
                {personal_context}
                
                Please generate 3 thoughtful reflection questions based on this quote and the user's personal context."""
            }
            
            payload = {
                **LLM_CONFIG,
                "messages": [system_message, user_message],
                "max_tokens": 200,  # Limit tokens for concise questions
                "temperature": 0.7  # Some creativity but not too random
            }
            
            response = requests.post(GROQ_API_URL, headers=HEADERS, json=payload)
            
            if response.status_code != 200:
                logger.error(f"Groq API error for questions: {response.text}")
                # Fallback to manual questions if AI fails
                return self._generate_fallback_questions(quote)
            
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"].strip()
            
            # Parse the response into individual questions
            questions = [q.strip() for q in ai_response.split('\n') if q.strip()]
            
            # Ensure we have exactly 3 questions
            while len(questions) < 3:
                questions.extend(self._generate_fallback_questions(quote)[:3-len(questions)])
            
            return questions[:3]  # Take only first 3 questions
            
        except Exception as e:
            logger.error(f"Error generating AI questions: {str(e)}")
            # Fallback to manual questions if AI fails
            return self._generate_fallback_questions(quote)

def load_knowledge_base():
    with open("knowledge_base.md", "r") as f:
        return f.read()

def find_relevant_quotes(knowledge_base, keywords):
    # Simple keyword search; can be improved with embeddings/LLM
    quotes = []
    for line in knowledge_base.split('\n'):
        if any(keyword.lower() in line.lower() for keyword in keywords):
            quotes.append(line)
    return random.sample(quotes, min(2, len(quotes)))  # Return up to 2 quotes

def generate_reflective_questions(values, feelings, challenges):
    questions = []
    if values:
        questions.append(f"What does '{values[0]}' mean to you today?")
    if feelings:
        questions.append(f"How does feeling '{feelings[0]}' affect your actions?")
    if challenges:
        questions.append(f"What would help you with '{challenges[0]}'?")
    # Add more question logic as needed
    return questions

def process_user_input(values, feelings, challenges, reflections):
    knowledge_base = load_knowledge_base()
    keywords = values + feelings + challenges
    quotes = find_relevant_quotes(knowledge_base, keywords)
    questions = generate_reflective_questions(values, feelings, challenges)
    return {
        "quotes": quotes,
        "questions": questions
    }