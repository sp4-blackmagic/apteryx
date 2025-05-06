# Apteryx - Desktop Application Interface

This directory contains the Streamlit-based desktop application interface for the Apteryx project.

## Setup and Installation using uv

1.  **Install uv (if you haven't already):**
    *   On macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    *   On Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    *   Or: `pip install uv`

2.  **Create and activate a virtual environment:**
    ```bash
    # Navigate to the apteryx_ui directory
    cd path/to/apteryx_ui

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
    *(Note: `numpy`, `pandas`, and `plotly` are useful for data handling and visualization. You can add more as needed.)*

5.  **Prepare a placeholder image (optional but recommended for camera screen):**
    Create an `assets` folder in `apteryx_ui` and place a simple PNG image named `placeholder_image.png` inside it. This will be used for the camera preview and captured image display. If you don't have one, the app will show an error for images, or you can modify the code to use `st.empty()` or text placeholders.

6.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

    The application should open in your default web browser.