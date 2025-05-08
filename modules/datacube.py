import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
from spectral import get_rgb
import plotly.graph_objects as go
from skimage.filters import threshold_otsu

def show_datacube_page():
    st.header("🗂️ Datacube Viewer")
    st.markdown("---")
    st.subheader("3D Cube Explorer")

    # Use the last preview datacube if available, else fallback to a random kiwi
    cube = st.session_state.get('last_preview_cube', None)
    metadata = st.session_state.get('last_preview_metadata', None)

    if cube is None:
        st.info("No datacube available. Please visit the Camera Control page first.")
        return

    if len(cube.shape) != 3:
        st.error("Loaded data is not a 3D datacube.")
        return

    h, w, bands = cube.shape
    wavelengths = None
    if metadata and 'wavelength' in metadata:
        try:
            wavelengths = [float(w) for w in metadata['wavelength']]
        except Exception:
            wavelengths = None

    # Downsample for performance if needed
    max_dim = 64
    ds_cube = cube
    if max(ds_cube.shape) > max_dim:
        factors = [max(1, s // max_dim) for s in ds_cube.shape]
        ds_cube = ds_cube[::factors[0], ::factors[1], ::factors[2]]
    # 3D Cube
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
        width=500, height=500,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Band',
            aspectmode='cube',
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    st.plotly_chart(fig3d, use_container_width=True)

    st.markdown("---")
    st.subheader("Band Selection & Background Removal")
    band = st.slider("Select Band for Analysis", 0, bands-1, 0)
    band_img = np.squeeze(cube[:, :, band])
    if band_img.ndim != 2:
        st.error("Selected band is not 2D.")
        return

    # Background removal options
    st.markdown("**Background Removal**")
    method = st.radio("Choose background removal method:", ["Otsu (auto)", "Manual threshold"])
    if method == "Otsu (auto)":
        threshold = threshold_otsu(band_img)
        st.caption(f"Otsu threshold: {threshold:.2f}")
        user_threshold = threshold
    else:
        min_val, max_val = float(np.min(band_img)), float(np.max(band_img))
        user_threshold = st.slider("Manual threshold", min_val, max_val, float(np.percentile(band_img, 10)), step=0.01)
    mask = band_img > user_threshold

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Before (Raw Band)**")
        st.image(Image.fromarray(((band_img - band_img.min()) * (255.0 / (band_img.max() - band_img.min()))).astype(np.uint8), mode='L'), use_container_width=True)
        fig, ax = plt.subplots(figsize=(4,2))
        ax.hist(band_img.flatten(), bins=50, color='#1f77b4', alpha=0.7)
        ax.set_title(f"Histogram (Raw)")
        st.pyplot(fig)
    with col_after:
        st.markdown("**After (Background Removed)**")
        masked_img = np.where(mask, band_img, 0)
        st.image(Image.fromarray(((masked_img - masked_img[mask].min()) * (255.0 / (masked_img[mask].max() - masked_img[mask].min()+1e-6))).astype(np.uint8), mode='L'), use_container_width=True)
        fig, ax = plt.subplots(figsize=(4,2))
        ax.hist(masked_img[mask].flatten(), bins=50, color='#e377c2', alpha=0.7)
        ax.set_title(f"Histogram (Foreground Only)")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Advanced Visualization & Export")
    with st.expander("Advanced Visualization Options", expanded=False):
        auto_contrast = st.checkbox("Auto Contrast Stretch", value=True)
        show_rgb = st.checkbox("Show RGB Composite (using bands 0, 1, 2)", value=False)
        show_pseudo = st.checkbox("Show Pseudocolor Mapping", value=False)
        hist_eq = st.checkbox("Histogram Equalization", value=False)
        export_csv = st.checkbox("Export Current Band as CSV", value=False)

    # Normalize for display (foreground only)
    fg_pixels = band_img[mask]
    if auto_contrast and fg_pixels.size > 0:
        p2, p98 = np.percentile(fg_pixels, (2, 98))
        band_img_norm = np.clip((band_img - p2) * 255.0 / (p98 - p2), 0, 255)
    else:
        band_img_norm = ((band_img - band_img.min()) * (255.0 / (band_img.max() - band_img.min()+1e-6)))
    if hist_eq and fg_pixels.size > 0:
        hist, bins = np.histogram(fg_pixels.flatten(), 256, [0,256])
        cdf = hist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        band_img_norm = np.interp(band_img_norm.flatten(), bins[:-1], cdf).reshape(band_img_norm.shape)
    band_img_norm = band_img_norm.astype(np.uint8)
    # Pseudocolor
    if show_pseudo:
        pseudo_img = plt.cm.viridis(band_img_norm/255.0)
        pseudo_img = (pseudo_img[:, :, :3] * 255).astype(np.uint8)
        st.image(pseudo_img, caption=f"Band {band} (Pseudocolor, Foreground)", use_container_width=True)
    # Grayscale
    img = Image.fromarray(band_img_norm, mode='L')
    st.image(img, caption=f"Band {band} (Grayscale, Foreground)", use_container_width=True)
    # RGB composite
    if show_rgb and bands >= 3:
        rgb = get_rgb(cube, [0, 1, 2])
        rgb_img = (rgb * 255).astype(np.uint8)
        st.image(rgb_img, caption="RGB Composite (Bands 0,1,2)", use_container_width=True)
    # Show band metadata
    if wavelengths and band < len(wavelengths):
        st.info(f"Wavelength: {wavelengths[band]:.2f} nm")
    # Region statistics (foreground only)
    st.markdown("**Region Statistics (Foreground Only):**")
    if fg_pixels.size > 0:
        stats = {
            'Mean': np.mean(fg_pixels),
            'Std': np.std(fg_pixels),
            'Min': np.min(fg_pixels),
            'Max': np.max(fg_pixels)
        }
        st.table(stats)
    else:
        st.warning("No foreground pixels detected with current threshold.")
    # Download as PNG
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    st.download_button("Download Current Band as PNG", data=buf.getvalue(), file_name=f"band_{band}_fg.png", mime="image/png")
    # Export as CSV
    if export_csv:
        csv = band_img.astype(np.float32)
        csv_bytes = io.StringIO()
        np.savetxt(csv_bytes, csv, delimiter=",")
        st.download_button("Download Current Band as CSV", data=csv_bytes.getvalue(), file_name=f"band_{band}_fg.csv", mime="text/csv")

    st.markdown("""
    <div style='color: #888; font-size: 1.1em;'>
    <b>Instructions:</b><br>
    - The datacube shown here is the same as the one used in the Camera Control preview.<br>
    - The 3D Cube Explorer lets you interactively explore the datacube in 3D.<br>
    - Use the slider to select a band and view its grayscale image.<br>
    - Remove background using Otsu or manual thresholding.<br>
    - Visualizations and statistics are computed on the foreground only.<br>
    - Optionally, enable auto contrast, RGB, pseudocolor, histogram equalization, and export.<br>
    - You can download the current band as a PNG or CSV.<br>
    </div>
    """, unsafe_allow_html=True) 