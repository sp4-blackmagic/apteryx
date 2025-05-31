import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini API key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
    if not GEMINI_API_KEY:
        st.markdown("""
        <div style='border-radius:10px; background:#fff3f3; padding:20px; margin:20px 0; border:1px solid #ffcccc; text-align:center;'>
            <b>💬 Kiwi Assistant (Chatbot Disabled)</b><br>
            <span style='color:#d9534f;'>No Gemini API key found.</span><br>
            <span style='color:#888;'>To enable the chatbot, add your Gemini API key to a <code>.env</code> file as <code>GEMINI_API_KEY=your-key-here</code> and restart the app.</span>
        </div>
        """, unsafe_allow_html=True)
        return
    st.markdown("""
    <style>
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 20px 0;
    }
    .chat-header {
        text-align: center;
        font-size: 1.2em;
        margin-bottom: 20px;
        color: #1f77b4;
    }
    .user-message {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2ca02c;
    }
    .typing-animation {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid #2ca02c;
        border-radius: 50%;
        border-top-color: transparent;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .reset-button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .reset-button:hover {
        background-color: #ff3333;
    }
    </style>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-header"><b>💬 Kiwi Assistant</b> &nbsp;|&nbsp; Ask me about the app, code, or graphs!</div>', unsafe_allow_html=True)

    # Add reset button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Reset Chat", key="reset_chat"):
            st.session_state.chat_history = []
            st.rerun()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    model = get_gemini_model()
    context = get_codebase_context()

    # Display chat history with custom styling
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 <b>Kiwi Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input with custom styling
    user_input = st.chat_input("Ask me anything about Apteryx, the code, or the graphs...")

    if user_input and model:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        st.markdown(f'<div class="user-message">👤 <b>You:</b><br>{user_input}</div>', unsafe_allow_html=True)

        # Show typing animation
        typing_placeholder = st.empty()
        typing_placeholder.markdown('<div class="typing-animation"></div>', unsafe_allow_html=True)

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

        # Remove typing animation and show response
        typing_placeholder.empty()
        st.markdown(f'<div class="assistant-message">🤖 <b>Kiwi Assistant:</b><br>{answer}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({'role': 'assistant', 'content': answer}) 