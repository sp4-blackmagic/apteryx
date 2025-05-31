import streamlit as st
import numpy as np
import os
from spectral import envi
from skimage.filters import threshold_otsu
import plotly.graph_objs as go
from scipy.signal import savgol_filter
from scipy.fft import fft

def list_kiwi_cubes():
    # Find all .hdr/.bin pairs in data/showcase/kiwi
    data_dir = "data/showcase/kiwi"
    files = os.listdir(data_dir)
    hdrs = [f for f in files if f.endswith('.hdr')]
    pairs = []
    for hdr in hdrs:
        base = hdr[:-4]
        bin_file = base + '.bin'
        if bin_file in files:
            pairs.append((base, os.path.join(data_dir, hdr), os.path.join(data_dir, bin_file)))
    return pairs

def load_cube(hdr_path, bin_path):
    img = envi.open(hdr_path, bin_path)
    cube = img.load()
    metadata = getattr(img, 'metadata', None)
    return cube, metadata

def show_comparison_page():
    st.header("🔬 Cube Comparison")
    st.markdown("---")
    st.markdown("Select two or more kiwi cubes to compare their 3D structure and spectral features. You can choose background removal method for fair comparison.")
    pairs = list_kiwi_cubes()
    if len(pairs) < 2:
        st.warning("Not enough cubes in data/showcase/kiwi to compare.")
        return
    cube_labels = [p[0] for p in pairs]
    selected = st.multiselect("Select at least 2 cubes to compare:", cube_labels, default=cube_labels[:2])
    if len(selected) < 2:
        st.info("Please select at least two different cubes.")
        return
    # Background removal method
    st.markdown("**Background Removal Method for All Cubes**")
    method = st.radio("Choose background removal method:", ["Manual threshold (default 0)", "Otsu (auto)", "Compare Both"], index=0)
    user_threshold = 0.0
    if method == "Manual threshold (default 0)":
        user_threshold = st.number_input("Manual threshold value (applied to all cubes, band 0)", value=0.0, step=0.01)
    # Load cubes
    cubes = []
    for sel in selected:
        for base, hdr, binf in pairs:
            if base == sel:
                cube, metadata = load_cube(hdr, binf)
                cube = np.asarray(cube)
                cubes.append({'label': base, 'cube': cube, 'metadata': metadata})
    # 3D Cube Explorers
    st.subheader("3D Cube Explorers (Background Removed)")
    cols = st.columns(len(cubes))
    for i, cdict in enumerate(cubes):
        with cols[i]:
            st.markdown(f"**{cdict['label']}**")
            cube = cdict['cube']
            band_img = cube[:, :, 0]
            if method == "Manual threshold (default 0)":
                mask = band_img > user_threshold
                cube_fg = cube * mask[..., None]
            elif method == "Otsu (auto)":
                from skimage.filters import threshold_otsu
                thresh = threshold_otsu(band_img)
                mask = band_img > thresh
                cube_fg = cube * mask[..., None]
            else:  # Compare Both
                from skimage.filters import threshold_otsu
                mask_manual = band_img > user_threshold
                cube_fg_manual = cube * mask_manual[..., None]
                thresh = threshold_otsu(band_img)
                mask_otsu = band_img > thresh
                cube_fg_otsu = cube * mask_otsu[..., None]
                # Show both 3D cubes
                st.markdown("Manual threshold (0):")
                # Downsample for performance
                max_dim = 48
                ds_cube = cube_fg_manual
                if max(ds_cube.shape) > max_dim:
                    factors = [max(1, s // max_dim) for s in ds_cube.shape]
                    ds_cube = ds_cube[::factors[0], ::factors[1], ::factors[2]]
                fig3d = go.Figure(data=go.Volume(
                    x=np.repeat(np.arange(ds_cube.shape[1]), ds_cube.shape[0]*ds_cube.shape[2]),
                    y=np.tile(np.repeat(np.arange(ds_cube.shape[0]), ds_cube.shape[2]), ds_cube.shape[1]),
                    z=np.tile(np.arange(ds_cube.shape[2]), ds_cube.shape[0]*ds_cube.shape[1]),
                    value=ds_cube.transpose(1,0,2).flatten(),
                    opacity=0.1,
                    surface_count=10,
                    colorscale='Viridis',
                ))
                fig3d.update_layout(
                    width=350, height=350,
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Band',
                        aspectmode='cube',
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                st.plotly_chart(fig3d, use_container_width=True)
                st.markdown("Otsu threshold:")
                ds_cube = cube_fg_otsu
                if max(ds_cube.shape) > max_dim:
                    factors = [max(1, s // max_dim) for s in ds_cube.shape]
                    ds_cube = ds_cube[::factors[0], ::factors[1], ::factors[2]]
                fig3d = go.Figure(data=go.Volume(
                    x=np.repeat(np.arange(ds_cube.shape[1]), ds_cube.shape[0]*ds_cube.shape[2]),
                    y=np.tile(np.repeat(np.arange(ds_cube.shape[0]), ds_cube.shape[2]), ds_cube.shape[1]),
                    z=np.tile(np.arange(ds_cube.shape[2]), ds_cube.shape[0]*ds_cube.shape[1]),
                    value=ds_cube.transpose(1,0,2).flatten(),
                    opacity=0.1,
                    surface_count=10,
                    colorscale='Viridis',
                ))
                fig3d.update_layout(
                    width=350, height=350,
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Band',
                        aspectmode='cube',
                    ),
                    margin=dict(l=0, r=0, b=0, t=0)
                )
                st.plotly_chart(fig3d, use_container_width=True)
                continue
            # Downsample for performance
            max_dim = 48
            ds_cube = cube_fg
            if max(ds_cube.shape) > max_dim:
                factors = [max(1, s // max_dim) for s in ds_cube.shape]
                ds_cube = ds_cube[::factors[0], ::factors[1], ::factors[2]]
            fig3d = go.Figure(data=go.Volume(
                x=np.repeat(np.arange(ds_cube.shape[1]), ds_cube.shape[0]*ds_cube.shape[2]),
                y=np.tile(np.repeat(np.arange(ds_cube.shape[0]), ds_cube.shape[2]), ds_cube.shape[1]),
                z=np.tile(np.arange(ds_cube.shape[2]), ds_cube.shape[0]*ds_cube.shape[1]),
                value=ds_cube.transpose(1,0,2).flatten(),
                opacity=0.1,
                surface_count=10,
                colorscale='Viridis',
            ))
            fig3d.update_layout(
                width=350, height=350,
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Band',
                    aspectmode='cube',
                ),
                margin=dict(l=0, r=0, b=0, t=0)
            )
            st.plotly_chart(fig3d, use_container_width=True)
    # Data Visualization Comparison
    st.markdown("---")
    st.subheader("Spectral Feature Comparison (Foreground Only)")
    # Compute features for each cube
    avg_spectra = []
    deriv1s = []
    deriv2s = []
    crs = []
    fft_mags = []
    for cdict in cubes:
        cube = cdict['cube']
        metadata = cdict['metadata']
        band_img = cube[:, :, 0]
        if method == "Manual threshold (default 0)":
            mask = band_img > user_threshold
            cube_fg = cube * mask[..., None]
        elif method == "Otsu (auto)":
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(band_img)
            mask = band_img > thresh
            cube_fg = cube * mask[..., None]
        else:
            from skimage.filters import threshold_otsu
            mask_manual = band_img > user_threshold
            cube_fg_manual = cube * mask_manual[..., None]
            thresh = threshold_otsu(band_img)
            mask_otsu = band_img > thresh
            cube_fg_otsu = cube * mask_otsu[..., None]
            # Show both spectra
            avg_spectrum_raw = np.mean(cube.reshape(-1, cube.shape[2]), axis=0)
            avg_spectrum_manual = np.mean(cube_fg_manual.reshape(-1, cube.shape[2]), axis=0)
            avg_spectrum_otsu = np.mean(cube_fg_otsu.reshape(-1, cube.shape[2]), axis=0)
            wavelengths = None
            if metadata and 'wavelength' in metadata:
                try:
                    wavelengths = np.array([float(w) for w in metadata['wavelength']])
                except Exception:
                    wavelengths = None
            x = wavelengths if wavelengths is not None and len(wavelengths) == len(avg_spectrum_raw) else np.arange(len(avg_spectrum_raw))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_raw, mode='lines', name='Raw (with background)', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_manual, mode='lines', name=f'Manual (>{user_threshold:.2f})', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=x, y=avg_spectrum_otsu, mode='lines', name=f'Otsu (>{thresh:.2f})', line=dict(color='orange')))
            fig.update_layout(title=f"{cdict['label']}: Raw vs Manual vs Otsu", xaxis_title="Wavelength (nm)" if wavelengths is not None else "Band Index", yaxis_title="Reflectance")
            st.plotly_chart(fig, use_container_width=True)
            continue
        avg_spectrum = np.mean(cube_fg.reshape(-1, cube.shape[2]), axis=0)
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
        avg_spectra.append((cdict['label'], avg_spectrum, wavelengths))
        deriv1s.append((cdict['label'], deriv1, wavelengths))
        deriv2s.append((cdict['label'], deriv2, wavelengths))
        crs.append((cdict['label'], cr, wavelengths))
        fft_mags.append((cdict['label'], fft_mag, None))
    # Overlay plots using Plotly
    def overlay_plot(data_list, title, ylabel, latex_exp, x_label="Wavelength (nm)"):
        st.markdown(f"**{title}**")
        st.latex(latex_exp)
        fig = go.Figure()
        # Define a distinct color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        for i, (label, y, wl) in enumerate(data_list):
            x = wl if wl is not None and len(wl) == len(y) else np.arange(len(y))
            fig.add_trace(go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                name=label,
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))
        fig.update_layout(
            title=title, 
            xaxis_title=x_label, 
            yaxis_title=ylabel,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    overlay_plot(avg_spectra, "Average Spectrum", "Reflectance", r"\bar{S}_{fg}(\lambda) = \frac{1}{N_{fg}} \sum_{i \in fg} S_i(\lambda)")
    overlay_plot(deriv1s, "1st Derivative", "d(Reflectance)/d\lambda", r"S'(\lambda) = \frac{d\bar{S}_{fg}}{d\lambda}")
    overlay_plot(deriv2s, "2nd Derivative", "d^2(Reflectance)/d\lambda^2", r"S''(\lambda) = \frac{d^2\bar{S}_{fg}}{d\lambda^2}")
    overlay_plot(crs, "Continuum Removed", "Intensity", r"CR(\lambda) = \bar{S}_{fg}(\lambda) - \min(\bar{S}_{fg})")
    overlay_plot(fft_mags, "FFT Magnitudes", "Magnitude", r"|FFT(\bar{S}_{fg}(\lambda))|", x_label="Frequency Index") 