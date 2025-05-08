import streamlit as st
import numpy as np
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