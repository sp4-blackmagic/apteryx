import streamlit as st
import os
from modules.camera import show_camera_page
from modules.visualization import show_visualization_page
from modules.inference import show_inference_page
from modules.about import show_about_page
from modules.datacube import show_datacube_page
from modules.comparison import show_comparison_page
from modules.kiwi_benefits import show_kiwi_benefits_page
from modules.components import show_help_page, show_feedback_page, show_settings_page
from modules.chatbot import show_chatbot
import yaml

# --- Page Configuration ---

st.set_page_config(
    page_title="Apteryx",
    page_icon="🥝",  # Kiwi emoji for browser tab icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# =======
# Sidebar
# =======
st.sidebar.title("Apteryx 🥝")

# Initialize active_screen in session state if it doesn't exist
if 'active_screen' not in st.session_state:
    st.session_state.active_screen = "Camera Control" # Default screen

# Load default config
if 'default_config' not in st.session_state:
    try:
        with open('default_config.yaml', 'r') as f:
            st.session_state.default_config = yaml.safe_load(f)
    except Exception:
        st.session_state.default_config = {}

# ============
# Main Screens
# ============
main_screen_options = {
    "Camera Control": "📷 Camera Control",
    "Data Visualization": "📊 Data Visualization",
    "Datacube": "🗂️ Datacube Viewer",
    "Comparison": "🔬 Cube Comparison",
    "Inference Engine": "🧠 Inference Engine",
    "Kiwi Benefits": "🥝 Why Eat More Kiwi?",
    "About": "👥 About Us"
}

# ===============
# Utility Screens
# ===============
utility_screen_options = {
    "Help": "❓ Help & Documentation",
    "Feedback": "📧 Feedback",
    "Settings": "⚙️ Settings"
}

# Create buttons for main screens
st.sidebar.subheader("Main Screens")
for screen_key, screen_name in main_screen_options.items():
    if st.sidebar.button(screen_name, key=f"btn_{screen_key}", use_container_width=True):
        st.session_state.active_screen = screen_key

st.sidebar.markdown("---")
st.sidebar.subheader("Support")

# Camera status logic
camera_ready = st.session_state.get('current_image') is not None
camera_preview = st.session_state.get('preview_kiwi_cube') is not None
if camera_ready:
    cam_color = '#28d94c'  # green
    cam_text = 'Camera Ready'
elif camera_preview:
    cam_color = '#ffd600'  # yellow
    cam_text = 'Camera Initialized (No Image)'
else:
    cam_color = '#ff4b4b'  # red
    cam_text = 'Camera Not Ready'
# Pulsing dot CSS
st.sidebar.markdown(f'''
<style>
.pulse-dot {{
  height: 16px;
  width: 16px;
  background-color: {cam_color};
  border-radius: 50%;
  display: inline-block;
  margin-right: 8px;
  box-shadow: 0 0 0 rgba(40, 217, 76, 0.7);
  animation: pulse 1.2s infinite;
}}
@keyframes pulse {{
  0% {{ box-shadow: 0 0 0 0 {cam_color}; }}
  70% {{ box-shadow: 0 0 0 10px rgba(40, 217, 76, 0); }}
  100% {{ box-shadow: 0 0 0 0 {cam_color}; }}
}}
</style>
<div style='display: flex; align-items: center; justify-content: center; margin-bottom: 10px;'>
  <span class="pulse-dot"></span>
  <span style='font-weight: 500; color: #333; text-align: center;'>{cam_text}</span>
</div>
''', unsafe_allow_html=True)

# Create buttons for utility screens
for screen_key, screen_name in utility_screen_options.items():
    if st.sidebar.button(screen_name, key=f"btn_{screen_key}", use_container_width=True):
        st.session_state.active_screen = screen_key

# --- Screen Implementations ---
active_screen_to_display = st.session_state.active_screen

if active_screen_to_display == "Camera Control":
    show_camera_page()
elif active_screen_to_display == "Data Visualization":
    show_visualization_page()
elif active_screen_to_display == "Datacube":
    show_datacube_page()
elif active_screen_to_display == "Comparison":
    show_comparison_page()
elif active_screen_to_display == "Kiwi Benefits":
    show_kiwi_benefits_page()
elif active_screen_to_display == "Inference Engine":
    show_inference_page()
elif active_screen_to_display == "About":
    show_about_page()
elif active_screen_to_display == "Help":
    show_help_page()
elif active_screen_to_display == "Feedback":
    show_feedback_page()
elif active_screen_to_display == "Settings":
    show_settings_page()

def show_help_page():
    st.header("❓ Help & Documentation")
    st.markdown("---")
    st.subheader("Welcome to Apteryx!")
    st.write("""
    Apteryx helps researchers assess kiwi fruit ripeness and firmness using hyperspectral imaging.
    This application provides an interface to control the camera, visualize data, and run predictions.

    **Navigation:**
    Use the buttons on the left sidebar to switch between screens:
    - **Camera Control:** Configure camera settings, capture hyperspectral images.
    - **Data Visualization:** Upload existing hyperspectral data (.hdr/.raw) to view spectral plots and features.
    - **Datacube Viewer:** Explore individual bands of a hyperspectral datacube.
    - **Inference Engine:** Upload hyperspectral data to get ripeness and firmness predictions from the ML model.

    **File Formats:**
    The system primarily uses ENVI hyperspectral file formats:
    - `.hdr`: Header file containing metadata about the image.
    - `.raw`/`.bin`: Binary file containing the raw pixel data.
    Ensure both files are provided together when uploading.

    **Camera Settings (Camera Control):**
    - **Integration Time:** Sensor exposure duration.
    - **Width/Height:** Image dimensions in pixels.
    - **Estimates:** Rough RAM/file size based on settings.

    **Data Visualization Options:**
    - **Average Spectrum:** Mean reflectance per wavelength.
    - *(Other options are placeholders).*

    **Inference:**
    - **Ripeness:** Categorical (Unripe, Ripe, Overripe).
    - **Firmness:** Numerical (0-10).
    - **Confidence:** Model's prediction confidence.

    For more details, refer to the project's main documentation (link would go here).
    """)

def show_feedback_page():
    st.header("📧 Feedback")
    st.markdown("---")
    st.write("We value your input! Please share any feedback, bug reports, or feature requests below.")

    with st.form("feedback_form"):
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Comment", "Question"])
        feedback_email = st.text_input("Your Email (Optional)")
        feedback_text = st.text_area("Your Feedback:", height=150, placeholder="Describe your feedback here...")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if feedback_text:
                st.success("Thank you for your feedback! (Simulated submission)")
                print(f"Feedback Received:\nType: {feedback_type}\nEmail: {feedback_email}\nMessage: {feedback_text}")
            else:
                st.error("Please enter your feedback before submitting.")

# At the very end of the main app logic, after all page rendering:
show_chatbot()