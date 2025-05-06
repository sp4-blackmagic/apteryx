import streamlit as st
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import time
import os

# --- Page Configuration ---

st.set_page_config(
    page_title="Apteryx",
    page_icon="🥝",  # Kiwi emoji for browser tab icon
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions & Placeholders ---

# Placeholder for image paths (if you create an assets folder)
PLACEHOLDER_IMG_PATH = os.path.join("assets", "placeholder_image.png")

def check_placeholder_image():
    if not os.path.exists(PLACEHOLDER_IMG_PATH):
        # st.sidebar.warning(f"Placeholder image not found at {PLACEHOLDER_IMG_PATH}. Image displays might be broken.")
        return False
    return True

placeholder_image_exists = check_placeholder_image()

def generate_dummy_spectrum_data():
    """Generates dummy spectral data for plotting."""
    wavelengths = np.linspace(400, 900, 150)
    reflectance = np.random.rand(150) * 0.6 + 0.1
    reflectance = np.sort(reflectance)
    df = pd.DataFrame({'Wavelength (nm)': wavelengths, 'Reflectance': reflectance})
    return df

def estimate_file_sizes(width, height, integration_time):
    """Estimates RAM usage and file size. Very rough estimation."""
    num_bands = 200
    bytes_per_pixel_band = 2
    raw_size_bytes = width * height * num_bands * bytes_per_pixel_band
    hdr_size_bytes = 1024
    ram_usage_bytes = raw_size_bytes * 2
    raw_size_mb = raw_size_bytes / (1024 * 1024)
    ram_usage_mb = ram_usage_bytes / (1024 * 1024)
    return f"{ram_usage_mb:.2f} MB", f"{raw_size_mb:.2f} MB (RAW) + {hdr_size_bytes/1024:.2f} KB (HDR)"

# =======
# Sidebar
# =======
st.sidebar.title("Apteryx 🥝")
st.sidebar.markdown("---")

# Initialize active_screen in session state if it doesn't exist
if 'active_screen' not in st.session_state:
    st.session_state.active_screen = "Camera Control" # Default screen

# ============
# Main Screens
# ============
main_screen_options = {
    "Camera Control": "📷 Camera Control",
    "Data Visualization": "📊 Data Visualization",
    "Inference Engine": "🧠 Inference Engine"
}

# ===============
# Utility Screens
# ===============
utility_screen_options = {
    "Help": "❓ Help & Documentation",
    "Feedback": "📧 Feedback"
}

# Create buttons for main screens
st.sidebar.subheader("Main Screens")
for screen_key, screen_name in main_screen_options.items():
    if st.sidebar.button(screen_name, key=f"btn_{screen_key}", use_container_width=True):
        st.session_state.active_screen = screen_key

st.sidebar.markdown("---")
st.sidebar.subheader("Support")
# Create buttons for utility screens
for screen_key, screen_name in utility_screen_options.items():
    if st.sidebar.button(screen_name, key=f"btn_{screen_key}", use_container_width=True):
        st.session_state.active_screen = screen_key


# --- Screen Implementations ---

# ==============
# Camera Control
# ==============
def screen_camera_control():
    st.header("📷 Camera Control")
    st.markdown("---")

    col_main, col_settings = st.columns([2, 1])

    with col_main:
        st.subheader("Camera Preview")
        preview_area = st.empty()
        if placeholder_image_exists:
            preview_area.image(PLACEHOLDER_IMG_PATH, caption="Live Camera Feed (Placeholder)", use_column_width=True)
        else:
            preview_area.markdown("<div style='height:300px; background-color:#f0f0f0; display:flex; align-items:center; justify-content:center; border:1px solid #ccc; margin-bottom: 1rem;'>Camera Preview Area</div>", unsafe_allow_html=True)


        if st.button("📸 Take Picture", key="take_picture_btn", type="primary"):
            st.session_state.picture_taken = True
            st.success("Picture captured! (Simulated)")

        if 'picture_taken' in st.session_state and st.session_state.picture_taken:
            st.subheader("Captured Image")
            if placeholder_image_exists:
                st.image(PLACEHOLDER_IMG_PATH, caption="Captured Hyperspectral Image (Placeholder)", use_column_width=True)
            else:
                st.markdown("<div style='height:300px; background-color:#e0e0e0; display:flex; align-items:center; justify-content:center; border:1px solid #aaa; margin-bottom: 1rem'>Captured Image Area</div>", unsafe_allow_html=True)

            dummy_hdr_content = "ENVI\ndescription = {Hyperspectral Image Header}\n"
            dummy_hdr_content += f"samples = {st.session_state.get('cam_width', 640)}\n"
            dummy_hdr_content += f"lines   = {st.session_state.get('cam_height', 480)}\n"
            dummy_hdr_content += "bands   = 200\ndata type = 12\ninterleave = bil\nsensor type = Unknown\nbyte order = 0\n"
            dummy_hdr_content += "wavelength = {400.000000, 402.500000, ..., 900.000000}"

            dummy_raw_content = np.random.bytes(st.session_state.get('cam_width', 640) * st.session_state.get('cam_height', 480) * 200 * 2)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="⬇️ Download .hdr",
                    data=dummy_hdr_content.encode('utf-8'),
                    file_name="captured_image.hdr",
                    mime="text/plain"
                )
            with col_dl2:
                st.download_button(
                    label="⬇️ Download .raw",
                    data=dummy_raw_content,
                    file_name="captured_image.raw",
                    mime="application/octet-stream"
                )

    with col_settings:
        st.subheader("⚙️ Camera Configuration")
        st.session_state.integration_time = st.number_input(
            "Integration Time (ms)",
            min_value=1,
            max_value=1000,
            value=st.session_state.get('integration_time', 100),
            step=1
        )
        st.session_state.cam_width = st.number_input(
            "Width (pixels)",
            min_value=100,
            max_value=4096,
            value=st.session_state.get('cam_width', 640),
            step=11
        )
        st.session_state.cam_height = st.number_input(
            "Height (pixels)",
            min_value=100,
            max_value=4096,
            value=st.session_state.get('cam_height', 480),
            step=11
        )

        st.markdown("---")
        st.markdown("**Estimates:**")
        ram_usage, file_size = estimate_file_sizes(
            st.session_state.cam_width,
            st.session_state.cam_height,
            st.session_state.integration_time
        )
        st.text(f"Expected RAM Usage: {ram_usage}")
        st.text(f"Expected File Size: {file_size}")

        if st.button("Apply Settings"):
            st.success("Camera settings applied (Simulated)")

# ==================
# Data Visualization
# ==================
def screen_data_visualization():
    st.header("📊 Data Visualization")
    st.markdown("---")

    st.subheader("Upload Hyperspectral Data")
    col_upload1, col_upload2 = st.columns(2)
    hdr_file_viz = col_upload1.file_uploader("Upload .hdr file", type="hdr", key="viz_hdr")
    raw_file_viz = col_upload2.file_uploader("Upload .raw file (corresponding to .hdr)", type="raw", key="viz_raw")

    if hdr_file_viz and raw_file_viz:
        st.success(f"Files '{hdr_file_viz.name}' and '{raw_file_viz.name}' uploaded successfully!")
        st.markdown("---")
        st.subheader("Spectral Reflectance Chart")

        chart_options = [
            "Average Spectrum",
            "Statistical Features",
            "Spectral Derivatives",
            "Continuum Spectrum",
            "Continuum Removal",
            "FFT Features"
        ]
        selected_chart_type = st.selectbox("Select Chart/Feature Type:", chart_options)

        if selected_chart_type == "Average Spectrum":
            df_spectrum = generate_dummy_spectrum_data()
            fig = px.line(df_spectrum, x='Wavelength (nm)', y='Reflectance', title='Average Reflectance Spectrum')
            fig.update_layout(yaxis_title='Reflectance (0-1)')
            st.plotly_chart(fig, use_container_width=True)
        elif selected_chart_type == "Statistical Features":
            st.info("Placeholder: Displaying statistical features of the spectrum.")
            df_spectrum = generate_dummy_spectrum_data()
            stats = df_spectrum['Reflectance'].agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
            stats.columns = ['Statistic', 'Value']
            st.table(stats)
        else:
            st.info(f"Placeholder: Visualization for '{selected_chart_type}' would be shown here.")

    elif hdr_file_viz or raw_file_viz:
        st.warning("Please upload both the .hdr and .raw files.")

# =========
# Inference
# =========
def screen_inference_engine():
    st.header("🧠 Inference Engine")
    st.markdown("---")

    st.subheader("Upload Hyperspectral Data for Prediction")
    col_upload_inf1, col_upload_inf2 = st.columns(2)
    hdr_file_inf = col_upload_inf1.file_uploader("Upload .hdr file", type="hdr", key="inf_hdr")
    raw_file_inf = col_upload_inf2.file_uploader("Upload .raw file (corresponding to .hdr)", type="raw", key="inf_raw")

    if hdr_file_inf and raw_file_inf:
        st.success(f"Files '{hdr_file_inf.name}' and '{raw_file_inf.name}' ready for inference.")

        if st.button("🥝 Predict Ripeness & Firmness", type="primary", key="predict_btn"):
            with st.spinner("Running inference... (Simulated)"):
                start_time = time.time()
                time.sleep(np.random.uniform(0.5, 2.0))
                end_time = time.time()
                prediction_time = end_time - start_time

                st.session_state.ripeness = np.random.choice(["Unripe", "Ripe", "Overripe"], p=[0.3, 0.5, 0.2])
                st.session_state.firmness = round(np.random.uniform(1.0, 9.5), 1)
                st.session_state.confidence = round(np.random.uniform(75.0, 99.0), 1)
                st.session_state.pred_time = f"{prediction_time:.2f} seconds"
                st.session_state.prediction_done = True

    elif hdr_file_inf or raw_file_inf:
        st.warning("Please upload both the .hdr and .raw files for inference.")


    if 'prediction_done' in st.session_state and st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("Prediction Results")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Ripeness Stage", st.session_state.ripeness)
        col_res2.metric("Firmness (Scale 0-10)", f"{st.session_state.firmness}")
        col_res3.metric("Confidence", f"{st.session_state.confidence}%")

        st.info(f"Time to generate prediction: {st.session_state.pred_time}")

# ===========
# Help Screen
# ===========
def screen_help():
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
    - **Inference Engine:** Upload hyperspectral data to get ripeness and firmness predictions from the ML model.

    **File Formats:**
    The system primarily uses ENVI hyperspectral file formats:
    - `.hdr`: Header file containing metadata about the image.
    - `.raw`: Binary file containing the raw pixel data.
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

# ===============
# Feedback Screen
# ===============
def screen_feedback():
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

# --- Main App Logic to Display Selected Screen ---
# The active screen is now controlled by st.session_state.active_screen,
# which is set by the sidebar buttons.
active_screen_to_display = st.session_state.active_screen

if active_screen_to_display == "Camera Control":
    screen_camera_control()
elif active_screen_to_display == "Data Visualization":
    screen_data_visualization()
elif active_screen_to_display == "Inference Engine":
    screen_inference_engine()
elif active_screen_to_display == "Help":
    screen_help()
elif active_screen_to_display == "Feedback":
    screen_feedback()