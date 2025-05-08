import streamlit as st
import numpy as np
from scipy.signal import savgol_filter
from scipy.fft import fft
from skimage.filters import threshold_otsu
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

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
        mask_sum = 0
        for b in range(cube.shape[2]):
            band_img = cube[:, :, b]
            try:
                thresh = threshold_otsu(band_img)
                mask = band_img > thresh
                cube_fg[:, :, b] = band_img * mask
                mask_all &= mask
                mask_sum += np.sum(mask)
            except Exception:
                cube_fg[:, :, b] = band_img
                mask_sum += np.sum(np.ones_like(band_img, dtype=bool))
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
        # Check if Otsu mask is all True or nearly all True
        total_pixels = cube.shape[0] * cube.shape[1] * cube.shape[2]
        if mask_sum / total_pixels > 0.98:
            st.markdown("**Average Spectrum (Foreground Only, Otsu did not remove background)**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_fg, mode='lines', name='Foreground Only (Otsu)', line=dict(color='orange', width=3)))
            fig.update_layout(title="Average Spectrum (Foreground Only)", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Reflectance")
            st.plotly_chart(fig, use_container_width=True)
            st.info("Otsu thresholding did not remove any significant background. Only the foreground spectrum is shown.")
        else:
            st.markdown("**Average Spectrum (Before/After Background Removal)**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_raw, mode='lines', name='Raw (with background)', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_fg, mode='lines', name='Foreground Only (Otsu)', line=dict(color='orange', width=3)))
            fig.update_layout(title="Average Spectrum Comparison", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Reflectance")
            st.plotly_chart(fig, use_container_width=True)
        st.latex(r"\bar{S}_{fg}(\lambda) = \frac{1}{N_{fg}} \sum_{i \in fg} S_i(\lambda)")
        # All further features use only foreground
        avg_spectrum = avg_spectrum_fg
        deriv1 = savgol_filter(avg_spectrum, 7, 2, deriv=1)
        deriv2 = savgol_filter(avg_spectrum, 7, 2, deriv=2)
        cr = avg_spectrum - np.min(avg_spectrum)
        fft_mag = np.abs(fft(avg_spectrum))
        # --- Explanations and Interactive Plots ---
        st.markdown("**1st Derivative**: The rate of change of the spectrum, highlighting edges and transitions.")
        st.latex(r"S'(\lambda) = \frac{d\bar{S}_{fg}}{d\lambda}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=deriv1, mode='lines', name="1st Derivative", line=dict(color='green')))
        fig.update_layout(title="1st Derivative (Foreground Only)", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="d(Reflectance)/d\lambda")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**2nd Derivative**: The curvature of the spectrum, useful for detecting peaks and valleys.")
        st.latex(r"S''(\lambda) = \frac{d^2\bar{S}_{fg}}{d\lambda^2}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=deriv2, mode='lines', name="2nd Derivative", line=dict(color='purple')))
        fig.update_layout(title="2nd Derivative (Foreground Only)", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="d^2(Reflectance)/d\lambda^2")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Continuum Removed**: The spectrum with its minimum value subtracted, emphasizing relative features.")
        st.latex(r"CR(\lambda) = \bar{S}_{fg}(\lambda) - \min(\bar{S}_{fg})")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=cr, mode='lines', name="Continuum Removed", line=dict(color='red')))
        fig.update_layout(title="Continuum Removed (Foreground Only)", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Intensity")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**FFT Magnitudes**: The magnitude of the Fast Fourier Transform of the average spectrum, showing frequency components.")
        st.latex(r"|FFT(\bar{S}_{fg}(\lambda))|")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.arange(len(fft_mag)), y=fft_mag, mode='lines', name="FFT Magnitudes", line=dict(color='blue')))
        fig.update_layout(title="FFT Magnitudes (Foreground Only)", xaxis_title="Frequency Index", yaxis_title="Magnitude")
        st.plotly_chart(fig, use_container_width=True)
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