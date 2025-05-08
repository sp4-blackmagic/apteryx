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
        st.subheader("Visualization of Last Captured Kiwi (Background Removal Comparison)")
        cube = np.asarray(captured_cube)
        metadata = captured_metadata
        # --- Background removal options ---
        st.markdown("**Background Removal Modes to Display**")
        show_raw = st.checkbox("Raw (no background removal)", value=True)
        show_manual = st.checkbox("Manual threshold", value=True)
        show_otsu = st.checkbox("Otsu (auto)", value=True)
        band_idx = 0  # Use band 0 for thresholding as a reference
        band_img = cube[:, :, band_idx]
        min_val, max_val = float(np.min(band_img)), float(np.max(band_img))
        user_threshold = st.slider("Manual threshold value", min_val, max_val, float(0), step=0.01) if show_manual else 0.0
        # Compute masks and foreground cubes
        mask_manual = band_img > user_threshold if show_manual else None
        mask_otsu = band_img > threshold_otsu(band_img) if show_otsu else None
        cube_fg_manual = cube * mask_manual[..., None] if show_manual else None
        cube_fg_otsu = cube * mask_otsu[..., None] if show_otsu else None
        # Compute spectra
        avg_spectrum_raw = np.mean(cube.reshape(-1, cube.shape[2]), axis=0) if show_raw else None
        avg_spectrum_manual = np.mean(cube_fg_manual.reshape(-1, cube.shape[2]), axis=0) if show_manual else None
        avg_spectrum_otsu = np.mean(cube_fg_otsu.reshape(-1, cube.shape[2]), axis=0) if show_otsu else None
        wavelengths = None
        if metadata and 'wavelength' in metadata:
            try:
                wavelengths = np.array([float(w) for w in metadata['wavelength']])
            except Exception:
                wavelengths = None
        x = wavelengths if wavelengths is not None and ((show_raw and len(wavelengths) == len(avg_spectrum_raw)) or (show_manual and len(wavelengths) == len(avg_spectrum_manual)) or (show_otsu and len(wavelengths) == len(avg_spectrum_otsu))) else np.arange(cube.shape[2])
        # --- Average Spectrum ---
        fig = go.Figure()
        if show_raw:
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_raw, mode='lines', name='Raw (with background)', line=dict(color='lightblue')))
        if show_manual:
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_manual, mode='lines', name=f'Manual (>{user_threshold:.2f})', line=dict(color='green')))
        if show_otsu:
            otsu_thr = threshold_otsu(band_img)
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_otsu, mode='lines', name=f'Otsu (>{otsu_thr:.2f})', line=dict(color='orange')))
        fig.update_layout(title="Average Spectrum: Raw vs Manual vs Otsu", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Reflectance")
        st.plotly_chart(fig, use_container_width=True)
        st.latex(r"\bar{S}_{fg}(\lambda) = \frac{1}{N_{fg}} \sum_{i \in fg} S_i(\lambda)")
        # --- 1st Derivative ---
        fig = go.Figure()
        if show_raw:
            deriv1 = savgol_filter(avg_spectrum_raw, 7, 2, deriv=1)
            fig.add_trace(go.Scatter(x=x, y=deriv1, mode='lines', name="1st Derivative (Raw)", line=dict(color='lightblue', dash='dot')))
        if show_manual:
            deriv1 = savgol_filter(avg_spectrum_manual, 7, 2, deriv=1)
            fig.add_trace(go.Scatter(x=x, y=deriv1, mode='lines', name="1st Derivative (Manual)", line=dict(color='green')))
        if show_otsu:
            deriv1 = savgol_filter(avg_spectrum_otsu, 7, 2, deriv=1)
            fig.add_trace(go.Scatter(x=x, y=deriv1, mode='lines', name="1st Derivative (Otsu)", line=dict(color='orange')))
        fig.update_layout(title="1st Derivative: Raw vs Manual vs Otsu", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="d(Reflectance)/d\lambda")
        st.latex(r"S'(\lambda) = \frac{d\bar{S}_{fg}}{d\lambda}")
        st.plotly_chart(fig, use_container_width=True)
        # --- 2nd Derivative ---
        fig = go.Figure()
        if show_raw:
            deriv2 = savgol_filter(avg_spectrum_raw, 7, 2, deriv=2)
            fig.add_trace(go.Scatter(x=x, y=deriv2, mode='lines', name="2nd Derivative (Raw)", line=dict(color='lightblue', dash='dot')))
        if show_manual:
            deriv2 = savgol_filter(avg_spectrum_manual, 7, 2, deriv=2)
            fig.add_trace(go.Scatter(x=x, y=deriv2, mode='lines', name="2nd Derivative (Manual)", line=dict(color='green')))
        if show_otsu:
            deriv2 = savgol_filter(avg_spectrum_otsu, 7, 2, deriv=2)
            fig.add_trace(go.Scatter(x=x, y=deriv2, mode='lines', name="2nd Derivative (Otsu)", line=dict(color='orange')))
        fig.update_layout(title="2nd Derivative: Raw vs Manual vs Otsu", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="d^2(Reflectance)/d\lambda^2")
        st.latex(r"S''(\lambda) = \frac{d^2\bar{S}_{fg}}{d\lambda^2}")
        st.plotly_chart(fig, use_container_width=True)
        # --- Continuum Removed ---
        fig = go.Figure()
        if show_raw:
            cr = avg_spectrum_raw - np.min(avg_spectrum_raw)
            fig.add_trace(go.Scatter(x=x, y=cr, mode='lines', name="Continuum Removed (Raw)", line=dict(color='lightblue', dash='dot')))
        if show_manual:
            cr = avg_spectrum_manual - np.min(avg_spectrum_manual)
            fig.add_trace(go.Scatter(x=x, y=cr, mode='lines', name="Continuum Removed (Manual)", line=dict(color='green')))
        if show_otsu:
            cr = avg_spectrum_otsu - np.min(avg_spectrum_otsu)
            fig.add_trace(go.Scatter(x=x, y=cr, mode='lines', name="Continuum Removed (Otsu)", line=dict(color='orange')))
        fig.update_layout(title="Continuum Removed: Raw vs Manual vs Otsu", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Intensity")
        st.latex(r"CR(\lambda) = \bar{S}_{fg}(\lambda) - \min(\bar{S}_{fg})")
        st.plotly_chart(fig, use_container_width=True)
        # --- FFT Magnitudes ---
        fig = go.Figure()
        if show_raw:
            fft_mag = np.abs(fft(avg_spectrum_raw))
            fig.add_trace(go.Scatter(x=np.arange(len(fft_mag)), y=fft_mag, mode='lines', name="FFT Magnitudes (Raw)", line=dict(color='lightblue', dash='dot')))
        if show_manual:
            fft_mag = np.abs(fft(avg_spectrum_manual))
            fig.add_trace(go.Scatter(x=np.arange(len(fft_mag)), y=fft_mag, mode='lines', name="FFT Magnitudes (Manual)", line=dict(color='green')))
        if show_otsu:
            fft_mag = np.abs(fft(avg_spectrum_otsu))
            fig.add_trace(go.Scatter(x=np.arange(len(fft_mag)), y=fft_mag, mode='lines', name="FFT Magnitudes (Otsu)", line=dict(color='orange')))
        fig.update_layout(title="FFT Magnitudes: Raw vs Manual vs Otsu", xaxis_title="Frequency Index", yaxis_title="Magnitude")
        st.latex(r"|FFT(\bar{S}_{fg}(\lambda))|")
        st.plotly_chart(fig, use_container_width=True)
        st.info("Data visualization generated for the last captured kiwi with selected background removal modes.")
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