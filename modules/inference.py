import streamlit as st
import numpy as np
import time

def show_inference_page():
    st.header("🧠 Inference Engine")
    st.markdown("---")

    st.subheader("Upload Hyperspectral Data for Prediction")
    col_upload_inf1, col_upload_inf2 = st.columns(2)
    hdr_file_inf = col_upload_inf1.file_uploader("Upload .hdr file", type="hdr", key="inf_hdr")
    raw_file_inf = col_upload_inf2.file_uploader("Upload .raw file (corresponding to .hdr)", type="raw", key="inf_raw")

    if hdr_file_inf and raw_file_inf:
        st.success(f"Files '{hdr_file_inf.name}' and '{raw_file_inf.name}' ready for inference.")

        if st.button("🥝 Predict Ripeness & Firmness", type="primary", key="predict_btn"):
            with st.spinner("Running inference... (Simulated)"):
                start_time = time.time()
                time.sleep(np.random.uniform(0.5, 2.0))
                end_time = time.time()
                prediction_time = end_time - start_time

                st.session_state.ripeness = np.random.choice(["Unripe", "Ripe", "Overripe"], p=[0.3, 0.5, 0.2])
                st.session_state.firmness = round(np.random.uniform(1.0, 9.5), 1)
                st.session_state.confidence = round(np.random.uniform(75.0, 99.0), 1)
                st.session_state.pred_time = f"{prediction_time:.2f} seconds"
                st.session_state.prediction_done = True

    elif hdr_file_inf or raw_file_inf:
        st.warning("Please upload both the .hdr and .raw files for inference.")

    if 'prediction_done' in st.session_state and st.session_state.prediction_done:
        st.markdown("---")
        st.subheader("Prediction Results")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Ripeness Stage", st.session_state.ripeness)
        col_res2.metric("Firmness (Scale 0-10)", f"{st.session_state.firmness}")
        col_res3.metric("Confidence", f"{st.session_state.confidence}%")

        st.info(f"Time to generate prediction: {st.session_state.pred_time}") 