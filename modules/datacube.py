import streamlit as st
import numpy as np
from spectral import envi
from PIL import Image
import os

def show_datacube_page():
    st.header("🗂️ Datacube Viewer")
    st.markdown("---")
    st.subheader("Upload Hyperspectral Datacube (.hdr + .bin/.raw)")
    col1, col2 = st.columns(2)
    hdr_file = col1.file_uploader("Upload .hdr file", type="hdr", key="datacube_hdr")
    bin_file = col2.file_uploader("Upload .bin or .raw file", type=["bin", "raw"], key="datacube_bin")

    if hdr_file and bin_file:
        # Save uploaded files to a temp location
        with open("temp_datacube.hdr", "wb") as f:
            f.write(hdr_file.read())
        with open("temp_datacube.bin", "wb") as f:
            f.write(bin_file.read())
        try:
            img = envi.open("temp_datacube.hdr", "temp_datacube.bin")
            cube = img.load()
            st.success(f"Loaded datacube with shape: {cube.shape}")
            if len(cube.shape) == 3:
                h, w, bands = cube.shape
                band = st.slider("Select Band", 0, bands-1, 0)
                band_img = cube[:, :, band]
                # Normalize for display
                band_img_norm = ((band_img - band_img.min()) * (255.0 / (band_img.max() - band_img.min()))).astype(np.uint8)
                st.image(band_img_norm, caption=f"Band {band} (Grayscale)", use_column_width=True, channels="GRAY")
            else:
                st.error("Loaded data is not a 3D datacube.")
        except Exception as e:
            st.error(f"Failed to load datacube: {e}")
        finally:
            os.remove("temp_datacube.hdr")
            os.remove("temp_datacube.bin")
    elif hdr_file or bin_file:
        st.info("Please upload both .hdr and .bin/.raw files.") 