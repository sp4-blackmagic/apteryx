import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.fft import fft
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
        st.subheader("Visualization of Last Captured Kiwi")
        cube = captured_cube
        metadata = captured_metadata
        avg_spectrum = np.mean(cube.reshape(-1, cube.shape[2]), axis=0)
        deriv1 = savgol_filter(avg_spectrum, 7, 2, deriv=1)
        deriv2 = savgol_filter(avg_spectrum, 7, 2, deriv=2)
        cr = avg_spectrum - np.min(avg_spectrum)
        fft_mag = np.abs(fft(avg_spectrum))
        wavelengths = None
        if metadata and 'wavelength' in metadata:
            try:
                wavelengths = np.array([float(w) for w in metadata['wavelength']])
            except Exception:
                wavelengths = None
        x = wavelengths if wavelengths is not None and len(wavelengths) == len(avg_spectrum) else np.arange(len(avg_spectrum))
        # --- Explanations ---
        st.markdown("**Average Spectrum**: The mean reflectance for each band across all pixels.")
        st.latex(r"\bar{S}(\lambda) = \frac{1}{N} \sum_{i=1}^N S_i(\lambda)")
        fig, ax = plt.subplots()
        ax.plot(x, avg_spectrum)
        ax.set_title("Average Spectrum")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("Reflectance")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**1st Derivative**: The rate of change of the spectrum, highlighting edges and transitions.")
        st.latex(r"S'(\lambda) = \frac{d\bar{S}}{d\lambda}")
        fig, ax = plt.subplots()
        ax.plot(x, deriv1)
        ax.set_title("1st Derivative")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("d(Reflectance)/d\lambda")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**2nd Derivative**: The curvature of the spectrum, useful for detecting peaks and valleys.")
        st.latex(r"S''(\lambda) = \frac{d^2\bar{S}}{d\lambda^2}")
        fig, ax = plt.subplots()
        ax.plot(x, deriv2)
        ax.set_title("2nd Derivative")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("d^2(Reflectance)/d\lambda^2")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**Continuum Removed**: The spectrum with its minimum value subtracted, emphasizing relative features.")
        st.latex(r"CR(\lambda) = \bar{S}(\lambda) - \min(\bar{S})")
        fig, ax = plt.subplots()
        ax.plot(x, cr)
        ax.set_title("Continuum Removed")
        ax.set_xlabel("Wavelength (nm)" if wavelengths is not None else "Band Index")
        ax.set_ylabel("Intensity")
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("**FFT Magnitudes**: The magnitude of the Fast Fourier Transform of the average spectrum, showing frequency components.")
        st.latex(r"|FFT(\bar{S}(\lambda))|")
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(fft_mag)), fft_mag)
        ax.set_title("FFT Magnitudes")
        ax.set_xlabel("Frequency Index")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        st.pyplot(fig)
        st.info("Data visualization generated for the last captured kiwi.")
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
        # (You can add similar math explanations and plotting for uploaded data here)
        st.info("Visualization for uploaded data is not implemented in this demo.")
    elif hdr_file_viz or raw_file_viz:
        st.warning("Please upload both the .hdr and .raw files.") 