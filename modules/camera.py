import streamlit as st
import os
from utils import get_random_kiwi_image, load_spectral_data, spectral_to_rgb, get_average_rgb

def show_camera_page():
    st.header("📷 Camera Control")
    st.markdown("---")

    col_main, col_settings = st.columns([2, 1])

    with col_main:
        st.subheader("Camera Preview")
        preview_area = st.empty()
        if os.path.exists("assets/placeholder_image.png"):
            preview_area.image("assets/placeholder_image.png", caption="Live Camera Feed (Placeholder)", use_column_width=True)
        else:
            preview_area.markdown("<div style='height:300px; background-color:#f0f0f0; display:flex; align-items:center; justify-content:center; border:1px solid #ccc; margin-bottom: 1rem;'>Camera Preview Area</div>", unsafe_allow_html=True)

        if st.button("📸 Take Picture", key="take_picture_btn", type="primary"):
            try:
                # Get random kiwi image
                kiwi_file = get_random_kiwi_image()
                # Load spectral data
                spectral_data = load_spectral_data(kiwi_file)
                # Convert to RGB
                rgb_image = spectral_to_rgb(spectral_data.load())
                # Get average RGB
                avg_rgb = get_average_rgb(rgb_image)
                
                st.session_state.picture_taken = True
                st.session_state.current_image = rgb_image
                st.session_state.avg_rgb = avg_rgb
                st.success("Picture captured from showcase!")
            except Exception as e:
                st.error(f"Error capturing image: {str(e)}")

        if 'picture_taken' in st.session_state and st.session_state.picture_taken:
            st.subheader("Captured Image")
            if 'current_image' in st.session_state:
                st.image(st.session_state.current_image, caption="Captured Hyperspectral Image", use_column_width=True)
                
                if 'avg_rgb' in st.session_state:
                    rgb = st.session_state.avg_rgb
                    st.markdown(f"**Average RGB Values:** R: {rgb[0]}, G: {rgb[1]}, B: {rgb[2]}")
                    # Display color swatch
                    st.markdown(f'<div style="width:100px; height:100px; background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); border:1px solid #ccc;"></div>', unsafe_allow_html=True)

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