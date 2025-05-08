import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fft import fft
from skimage.filters import threshold_otsu
import pandas as pd
import plotly.express as px

def generate_dummy_spectrum_data():
    """Generates dummy spectral data for plotting."""
    wavelengths = np.linspace(400, 900, 150)
    reflectance = np.random.rand(150) * 0.6 + 0.1
    reflectance = np.sort(reflectance)
    df = pd.DataFrame({'Wavelength (nm)': wavelengths, 'Reflectance': reflectance})
    return df

def show_visualization_page():
    st.header("📊 Data Visualization")
    st.markdown("---")

    # If a kiwi has been captured, show its data visualization
    captured_cube = st.session_state.get('captured_cube')
    captured_metadata = st.session_state.get('captured_metadata')
    if captured_cube is not None:
        st.subheader("Visualization of Last Captured Kiwi (Background Removed)")
        cube = captured_cube
        metadata = captured_metadata
        # Remove background using Otsu for each band
        cube_fg = np.zeros_like(cube)
        mask_all = np.ones(cube.shape[:2], dtype=bool)
        for b in range(cube.shape[2]):
            band_img = cube[:, :, b]
            try:
                thresh = threshold_otsu(band_img)
                mask = band_img > thresh
                cube_fg[:, :, b] = band_img * mask
                mask_all &= mask
            except Exception:
                cube_fg[:, :, b] = band_img
        # Average spectrum (before/after)
        avg_spectrum_raw = np.mean(cube.reshape(-1, cube.shape[2]), axis=0)
        avg_spectrum_fg = np.mean(cube_fg.reshape(-1, cube.shape[2]), axis=0)
        wavelengths = None
        if metadata and 'wavelength' in metadata:
            try:
                wavelengths = np.array([float(w) for w in metadata['wavelength']])
            except Exception:
                wavelengths = None
        x = wavelengths if wavelengths is not None and len(wavelengths) == len(avg_spectrum_raw) else np.arange(len(avg_spectrum_raw))
        st.markdown("**Average Spectrum (Before/After Background Removal)**")
        fig, ax = plt.subplots()
        ax.plot(x, avg_spectrum_raw, label="Raw (with background)", alpha=0.5)
        ax.plot(x, avg_spectrum_fg, label="Foreground Only (Otsu)", linewidth=2)
        ax.set_title("Average Spectrum Comparison")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("Reflectance")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        st.latex(r"\bar{S}_{fg}(\lambda) = \frac{1}{N_{fg}} \sum_{i \in fg} S_i(\lambda)")
        # All further features use only foreground
        avg_spectrum = avg_spectrum_fg
        deriv1 = savgol_filter(avg_spectrum, 7, 2, deriv=1)
        deriv2 = savgol_filter(avg_spectrum, 7, 2, deriv=2)
        cr = avg_spectrum - np.min(avg_spectrum)
        fft_mag = np.abs(fft(avg_spectrum))
        # --- Explanations ---
        st.markdown("**1st Derivative**: The rate of change of the spectrum, highlighting edges and transitions.")
        st.latex(r"S'(\lambda) = \frac{d\bar{S}_{fg}}{d\lambda}")
        fig, ax = plt.subplots()
        ax.plot(x, deriv1)
        ax.set_title("1st Derivative (Foreground Only)")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("d(Reflectance)/d\lambda")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**2nd Derivative**: The curvature of the spectrum, useful for detecting peaks and valleys.")
        st.latex(r"S''(\lambda) = \frac{d^2\bar{S}_{fg}}{d\lambda^2}")
        fig, ax = plt.subplots()
        ax.plot(x, deriv2)
        ax.set_title("2nd Derivative (Foreground Only)")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("d^2(Reflectance)/d\lambda^2")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**Continuum Removed**: The spectrum with its minimum value subtracted, emphasizing relative features.")
        st.latex(r"CR(\lambda) = \bar{S}_{fg}(\lambda) - \min(\bar{S}_{fg})")
        fig, ax = plt.subplots()
        ax.plot(x, cr)
        ax.set_title("Continuum Removed (Foreground Only)")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**FFT Magnitudes**: The magnitude of the Fast Fourier Transform of the average spectrum, showing frequency components.")
        st.latex(r"|FFT(\bar{S}_{fg}(\lambda))|")
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(fft_mag)), fft_mag)
        ax.set_title("FFT Magnitudes (Foreground Only)")
        ax.set_xlabel("Frequency Index")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        st.pyplot(fig)
        st.info("Data visualization generated for the last captured kiwi (background removed with Otsu).")
        return

    # If no capture, allow upload as before
    st.subheader("Upload Hyperspectral Data")
    col_upload1, col_upload2 = st.columns(2)
    hdr_file_viz = col_upload1.file_uploader("Upload .hdr file", type="hdr", key="viz_hdr")
    raw_file_viz = col_upload2.file_uploader("Upload .raw file (corresponding to .hdr)", type="raw", key="viz_raw")

    if hdr_file_viz and raw_file_viz:
        st.success(f"Files '{hdr_file_viz.name}' and '{raw_file_viz.name}' uploaded successfully!")
        st.markdown("---")
        st.subheader("Spectral Reflectance Chart")
        st.info("Visualization for uploaded data is not implemented in this demo.")
    elif hdr_file_viz or raw_file_viz:
        st.warning("Please upload both the .hdr and .raw files.") 