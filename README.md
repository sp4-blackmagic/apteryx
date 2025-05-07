# Apteryx - Mock App for the Presentation

This is a simple mockup for the application interface using streamlit that was shamelessly vibecoded.

## Setup and Installation using uv

1.  **Install uv:**
    *   On macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   On Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    *   Or: `pip install uv`

2.  **Create and activate a virtual environment:**
    ```bash
    # In the project directory...
    # Create the virtual environment
    uv venv .venv
    ```

3.  **Activate the virtual environment:**
    *   Windows (PowerShell): `.venv\Scripts\Activate.ps1`
    *   Windows (Cmd.exe): `.venv\Scripts\activate.bat`
    *   Linux/macOS: `source .venv/bin/activate`

4.  **Install dependencies:**
    ```bash
    uv pip install streamlit numpy pandas plotly
    ```

5.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    The application should open in your default web browser.

6.  **Placeholder images:**

      By default, the app will display a `st.empty()` for places, where images should be (taken by camera, camera preview, and so on). If you want to change that, you should create an `assets` folder in the project root directory and place a PNG image named `placeholder_image.png` inside it.

