# Philosophical AI Assistant

A RAG-based AI assistant that provides philosophical insights while considering personal context and values. The system combines personal context awareness with a knowledge base of philosophical content to deliver meaningful, personalized responses.

## Features

- 🧠 **RAG-Based Responses**: Utilizes Retrieval-Augmented Generation for contextually relevant answers
- 🎯 **Personal Context Awareness**: Considers user's values, challenges, and goals
- 💭 **Philosophical Knowledge Base**: Draws from embedded philosophical texts
- 🔄 **Conversation Memory**: Maintains context through conversation history
- 🖥️ **User-Friendly Interface**: Streamlit-based frontend for easy interaction

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Store**: FAISS
- **LLM**: Groq
- **Embeddings**: LangChain
- **Database**: In-memory storage (with conversation persistence)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PhilosophicalLLM.git
cd PhilosophicalLLM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## Usage

1. Start the FastAPI backend:
```bash
uvicorn main:app --reload
```

2. In a new terminal, launch the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Open your browser and navigate to:
- Frontend: `http://localhost:8501`
- API docs: `http://localhost:8000/docs`

## Project Structure

```
PhilosophicalLLM/
├── App/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── model/
│   │   └── schemas.py
│   └── services/
│       └── chat_service.py
├── main.py
├── streamlit_app.py
├── knowledge_base.md
└── requirements.txt
```

## Configuration

Key configurations in `App/core/config.py`:
- `KNOWLEDGE_BASE_PATH`: Path to your philosophical knowledge base
- `CHUNK_SIZE`: Size of text chunks for embedding
- `CHUNK_OVERLAP`: Overlap between chunks
- `EMBEDDING_MODEL`: Model used for text embeddings

## API Endpoints

- `POST /api/update_context`: Update user's personal context
- `POST /api/ask`: Submit a question to the AI
- `GET /api/history`: Retrieve conversation history
- `POST /api/clear_history`: Clear conversation history

## Frontend Features

- Personal context management (values, challenges, goals)
- Interactive chat interface
- Conversation history viewing and management
- Real-time response display

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Groq LLM API
- Inspired by philosophical counseling and RAG-based systems
- Uses LangChain for embeddings and document processing