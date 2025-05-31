import streamlit as st
import os
import numpy as np
from utils import get_random_kiwi_image, load_spectral_data, spectral_to_rgb, get_average_rgb
from PIL import Image

def show_camera_page():
    st.header("📷 Camera Control")
    st.markdown("---")

    col_main, col_settings = st.columns([2, 1])

    # --- Camera Settings ---
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

    # --- Camera Preview & Capture ---
    with col_main:
        st.subheader("Camera Preview")
        preview_area = st.empty()
        # Only load a kiwi if not already loaded for this session
        if 'preview_kiwi_cube' not in st.session_state or st.session_state.get('preview_kiwi_cube') is None:
            try:
                kiwi_file = get_random_kiwi_image()
                spectral_data = load_spectral_data(kiwi_file)
                cube = spectral_data.load()
                st.session_state.preview_kiwi_cube = cube
                st.session_state.preview_kiwi_metadata = getattr(spectral_data, 'metadata', None)
            except Exception as e:
                preview_area.error(f"Could not load preview: {e}")
                st.session_state.preview_kiwi_cube = None
                st.session_state.preview_kiwi_metadata = None
        cube = st.session_state.get('preview_kiwi_cube')
        metadata = st.session_state.get('preview_kiwi_metadata')
        if cube is not None:
            # Use the first band for grayscale preview
            if len(cube.shape) == 3:
                band_img = cube[:, :, 0]
            elif len(cube.shape) == 2:
                band_img = cube
            else:
                preview_area.error("Unsupported datacube shape for preview.")
                band_img = None
            if band_img is not None:
                band_img = np.squeeze(band_img)
                if band_img.ndim == 2:
                    band_img_norm = ((band_img - band_img.min()) * (255.0 / (band_img.max() - band_img.min()))).astype(np.uint8)
                    img = Image.fromarray(band_img_norm, mode='L')
                    img = img.resize((int(st.session_state.cam_width), int(st.session_state.cam_height)), Image.BILINEAR)
                    preview_area.image(img, caption="Live Camera Feed (Grayscale Kiwi, Band 0)", use_container_width=True)
        else:
            preview_area.error("No preview available.")

        if st.button("📸 Take Picture", key="take_picture_btn", type="primary"):
            if cube is not None:
                # Use the current previewed kiwi for capture
                spectral_data = cube
                st.session_state.picture_taken = True
                st.session_state.captured_cube = spectral_data
                st.session_state.captured_metadata = metadata
                # Also update the datacube viewer
                st.session_state.last_preview_cube = spectral_data
                st.session_state.last_preview_metadata = metadata
                # Show RGB image after capture
                rgb_image = spectral_to_rgb(spectral_data)
                st.session_state.current_image = rgb_image
                st.success("Picture captured from preview!")
            else:
                st.error("No preview datacube to capture.")

        # --- Show RGB image after capture ---
        if st.session_state.get('picture_taken') and st.session_state.get('current_image') is not None:
            st.markdown("---")
            st.subheader("Captured RGB Image")
            st.image(st.session_state.current_image, caption="Captured Hyperspectral Image (RGB Approximation)", use_container_width=True)
            avg_rgb = get_average_rgb(st.session_state.current_image)
            if avg_rgb is not None:
                st.markdown(f"**Average RGB Values:** R: {avg_rgb[0]}, G: {avg_rgb[1]}, B: {avg_rgb[2]}")
                st.markdown(f'<div style="width:100px; height:100px; background-color:rgb({avg_rgb[0]},{avg_rgb[1]},{avg_rgb[2]}); border:1px solid #ccc;"></div>', unsafe_allow_html=True)

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