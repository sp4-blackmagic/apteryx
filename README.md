# Apteryx - Mock App for the Presentation

This is a simple mockup for the application interface using Streamlit.

## Quick Start

1. **Clone the repository and enter the project directory.**

2. **Run the setup script:**

   ```bash
   bash setup.sh
   ```

   This script will:
   - Create and activate a virtual environment (if not already present)
   - Install all required dependencies (using `uv` if available)
   - Launch the Streamlit application automatically

3. **Access the app:**

   The application should open in your default web browser. If not, visit the URL shown in your terminal (usually http://localhost:8501).

---

### Notes

- **Sample Data:**

  The repository includes sample data in the `data/` directory for demonstration and testing purposes. You can use these files to explore the app's features without needing to capture new data immediately.

- **Placeholder images:**

  By default, the app will display a `st.empty()` for places where images should be (taken by camera, camera preview, etc.). If you want to change that, create an `assets` folder in the project root directory and place a PNG image named `placeholder_image.png` inside it.

- **Requirements:**

  The `setup.sh` script will check for Python 3.8+, `uv`, and other dependencies. If something is missing, it will attempt to install it or prompt you with instructions.

- **Gemini API Key (for Chatbot, Optional):**

  If you want to use the built-in chatbot, you need a Gemini API key from Google. You can either:
  - Enter it when prompted by the setup script, **or**
  - Create a `.env` file in the project root with the following content:
    
    ```env
    GEMINI_API_KEY=your-gemini-api-key-here
    ```
  If you skip this step, the app will still work, but the chatbot will not be available.

---

For more details, see the comments in `