#!/bin/bash

# Welcome message
cat <<'EOF'
========================================
   Welcome to the Apteryx Setup Script!  
========================================
This script will set up your environment, install dependencies, and launch the app.
EOF

# Prompt for Gemini API key (demo only)
echo "\n[INFO] For demonstration: Please enter your Gemini API key (press Enter to skip)."
read -r -p "Gemini API Key: " GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    echo "[INFO] No Gemini API key provided. The chatbot will NOT be available."
else
    echo "[INFO] Saving Gemini API key to .env file."
    echo "GEMINI_API_KEY=$GEMINI_API_KEY" > .env
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print("{}.{}".format(sys.version_info[0], sys.version_info[1]))')
REQUIRED_VERSION="3.8"
if [[ $(echo -e "$PYTHON_VERSION\n$REQUIRED_VERSION" | sort -V | head -n1) != "$REQUIRED_VERSION" ]]; then
    echo "[ERROR] Python 3.8 or higher is required. You have $PYTHON_VERSION."
    exit 1
fi

# # Create virtual environment if it doesn't exist
# if [ ! -d ".venv" ]; then
#     echo "[INFO] Creating virtual environment..."
#     python3 -m venv .venv
# fi

# # Activate virtual environment
# source .venv/bin/activate

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "[INFO] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH if needed
    export PATH="$HOME/.local/bin:$PATH"
fi

# # Check for requirements.txt
# if [ ! -f "requirements.txt" ]; then
#     echo "[ERROR] requirements.txt not found! Please provide one."
#     exit 1
# fi

# Install dependencies using uv
# uv pip install -r requirements.txt

# # Check for streamlit
# if ! python -c "import streamlit" &> /dev/null; then
#     echo "[ERROR] Streamlit is not installed. Please check your requirements.txt."
#     exit 1
# fi

uv add -r requirements.txt

# Goodbye message
cat <<'EOF'
========================================
  Setup complete! Launching Apteryx...  
========================================
EOF

# Run the app
uv run streamlit run app.py 