import os
import streamlit as st
import requests
from typing import List

# Constants
API_BASE_URL = "http://localhost:8000/api"  # Adjust if your FastAPI runs on a different port or prefix

def update_personal_context(values: List[str], challenges: List[str], goals: List[str]) -> None:
    """Update the user's personal context."""
    context = {
        "values": values,
        "challenges": challenges,
        "goals": goals
    }
    try:
        response = requests.post(
            f"{API_BASE_URL}/update_context",
            json=context,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            st.success("Personal context updated successfully!")
        else:
            st.error(f"Error updating context: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")

def ask_question(question: str) -> None:
    """Send a question to the API and display the response."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            data={"question": question},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            st.write("🤖 Assistant:", response.json().get("answer", "No response"))
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")

def view_history() -> None:
    """Fetch and display chat history."""
    try:
        response = requests.get(f"{API_BASE_URL}/history")
        if response.status_code == 200:
            history = response.json()
            for msg in history:
                role = "🤖 Assistant:" if msg.get("role") == "assistant" else "👤 You:"
                st.write(f"{role} {msg.get('content', 'No content')}")
        else:
            st.error(f"Error fetching history: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")

def clear_history() -> None:
    """Clear chat history."""
    try:
        response = requests.post(f"{API_BASE_URL}/clear_history")
        if response.status_code == 200:
            st.success("Chat history cleared!")
        else:
            st.error(f"Error clearing history: {response.json().get('detail', 'Unknown error')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")

def main():
    st.set_page_config(page_title="Philosophical AI Assistant", page_icon="🤖")
    st.title("Philosophical AI Assistant")

    # Sidebar for personal context
    st.sidebar.header("Personal Context")
    
    values = st.sidebar.text_input("Values (comma-separated)", "growth, authenticity, compassion").split(",")
    values = [v.strip() for v in values if v.strip()]
    
    challenges = st.sidebar.text_input("Challenges (comma-separated)", "work-life balance, self-doubt").split(",")
    challenges = [c.strip() for c in challenges if c.strip()]
    
    goals = st.sidebar.text_input("Goals (comma-separated)", "self-improvement, meaningful work").split(",")
    goals = [g.strip() for g in goals if g.strip()]

    if st.sidebar.button("Update Context"):
        update_personal_context(values, challenges, goals)

    # Main chat interface
    st.subheader("Chat")
    question = st.text_input("Ask your question:")
    if st.button("Send"):
        if question:
            ask_question(question)
        else:
            st.warning("Please enter a question.")

    # History section
    st.subheader("Chat History")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View History"):
            view_history()
    with col2:
        if st.button("Clear History"):
            clear_history()

if __name__ == "__main__":
    main()