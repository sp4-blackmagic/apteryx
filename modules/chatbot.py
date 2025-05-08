import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("Gemini API key not found. Please set GEMINI_API_KEY in your .env file.")

def get_codebase_context():
    # Load all non-sensitive code files for context
    context = ""
    for root, dirs, files in os.walk("."):
        # Skip venv, .git, data, and other sensitive/large folders
        if any(skip in root for skip in ['.venv', '.git', 'data', '__pycache__']):
            continue
        for file in files:
            if file.endswith(('.py', '.md', '.yaml', '.yml')) and not file.startswith('.'):
                try:
                    with open(os.path.join(root, file), 'r') as f:
                        content = f.read()
                        context += f"\n# File: {os.path.join(root, file)}\n{content}\n"
                except Exception:
                    continue
    return context[:20000]  # Limit context size for API

@st.cache_resource
def get_gemini_model():
    if not GEMINI_API_KEY:
        return None
    return genai.GenerativeModel('gemini-1.5-flash')

def show_chatbot():
    st.markdown("---")
    st.markdown("<div style='text-align:center; font-size:1.2em;'><b>💬 Kiwi Assistant</b> &nbsp;|&nbsp; Ask me about the app, code, or graphs!</div>", unsafe_allow_html=True)
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    model = get_gemini_model()
    context = get_codebase_context()
    for msg in st.session_state.chat_history:
        st.chat_message(msg['role']).write(msg['content'])
    user_input = st.chat_input("Ask me anything about Apteryx, the code, or the graphs...")
    if user_input and model:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                prompt = f"""
You are Kiwi Assistant, a helpful AI for the Apteryx hyperspectral imaging app. You can:
- Explain any graph or feature in the app
- Help users understand the codebase (see context below)
- Answer questions about groups, comparison, and kiwi science

Codebase context:
{context}

User question: {user_input}
"""
                try:
                    response = model.generate_content(prompt)
                    answer = response.text
                except Exception as e:
                    answer = f"Sorry, I couldn't get an answer from Gemini: {e}"
                st.write(answer)
                st.session_state.chat_history.append({'role': 'assistant', 'content': answer}) 