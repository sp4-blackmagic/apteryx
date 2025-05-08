import streamlit as st
import numpy as np
from spectral import envi
import matplotlib.pyplot as plt
import tempfile
import os

def show_datacube_page():
    st.header("🗂️ Datacube Viewer")
    st.markdown("---")
    st.subheader("Upload Hyperspectral Datacube (.hdr + .bin/.raw)")
    col1, col2 = st.columns(2)
    hdr_file = col1.file_uploader("Upload .hdr file", type="hdr", key="datacube_hdr")
    bin_file = col2.file_uploader("Upload .bin or .raw file", type=["bin", "raw"], key="datacube_bin")

    if hdr_file and bin_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdr') as hdr_tmp, \
             tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as bin_tmp:
            hdr_tmp.write(hdr_file.read())
            bin_tmp.write(bin_file.read())
            hdr_path = hdr_tmp.name
            bin_path = bin_tmp.name
        try:
            img = envi.open(hdr_path, bin_path)
            cube = img.load()
            st.success(f"Loaded datacube with shape: {cube.shape}")
            if len(cube.shape) == 3:
                h, w, bands = cube.shape
                # Try to get wavelengths from metadata
                wavelengths = None
                if hasattr(img, 'metadata') and 'wavelength' in img.metadata:
                    try:
                        wavelengths = [float(w) for w in img.metadata['wavelength']]
                    except Exception:
                        wavelengths = None
                band = st.slider("Select Band", 0, bands-1, 0)
                band_img = cube[:, :, band]
                # Advanced options
                with st.expander("Advanced Visualization Options", expanded=False):
                    auto_contrast = st.checkbox("Auto Contrast Stretch", value=True)
                # Normalize for display
                if auto_contrast:
                    p2, p98 = np.percentile(band_img, (2, 98))
                    band_img_norm = np.clip((band_img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
                else:
                    band_img_norm = ((band_img - band_img.min()) * (255.0 / (band_img.max() - band_img.min()))).astype(np.uint8)
                st.image(band_img_norm, caption=f"Band {band} (Grayscale)", use_column_width=True, channels="GRAY")
                # Show band metadata
                if wavelengths and band < len(wavelengths):
                    st.info(f"Wavelength: {wavelengths[band]:.2f} nm")
                # Show histogram
                fig, ax = plt.subplots(figsize=(4,2))
                ax.hist(band_img.flatten(), bins=50, color='#1f77b4', alpha=0.7)
                ax.set_title(f"Histogram of Band {band}")
                ax.set_xlabel("Pixel Value")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.error("Loaded data is not a 3D datacube.")
        except Exception as e:
            st.error(f"Failed to load datacube: {e}")
        finally:
            os.remove(hdr_path)
            os.remove(bin_path)
    elif hdr_file or bin_file:
        st.info("Please upload both .hdr and .bin/.raw files.")
    else:
        st.markdown("""
        <div style='color: #888; font-size: 1.1em;'>
        <b>Instructions:</b><br>
        - Upload both the <b>.hdr</b> and <b>.bin/.raw</b> files for your hyperspectral datacube.<br>
        - Use the slider to select a band and view its grayscale image.<br>
        - Optionally, enable auto contrast for better visualization.<br>
        - The histogram below the image shows the distribution of pixel values for the selected band.<br>
        </div>
        """, unsafe_allow_html=True) 