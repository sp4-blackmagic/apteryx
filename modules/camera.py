import streamlit as st
import requests
from PIL import Image
import numpy as np
import io


def show_camera_page():
    st.header("📷 Camera Control")
    st.markdown("---")

    # Initialize session state variables
    if "connection_status" not in st.session_state:
        st.session_state.connection_status = "disconnected"
    if "config" not in st.session_state:
        st.session_state.config = {
            "cube_width": 500,
            "cube_height": 500,
            "integration_time_ms": 60,
            "wavelength_start_nm": 470,
            "wavelength_end_nm": 900,
            "preview": False,
        }
    st.success(st.session_state.connection_status)

    # --- Connection Section ---
    with st.expander("🔗 Camera Connection", expanded=True):
        camera_url = st.text_input(
            "Camera API URL",
            value=st.session_state.get("camera_url", ""),
            key="camera_url_input",
        )
        if camera_url:
            st.session_state.full_url = f"http://{camera_url}"

        # disabled=st.session_state.connection_status == "connected"
        if st.button("Connect"):
            try:
                response = requests.post(
                    f"{st.session_state.full_url}/connect", json={}
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "connected"
                    st.success("Successfully connected to camera!")
                else:
                    st.error(f"Connection failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")

        # disabled=st.session_state.connection_status == "disconnected
        if st.button("Disconnect"):
            try:
                response = requests.post(
                    f"{st.session_state.full_url}/disconnect", json={}
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "disconnected"
                    st.success("Successfully disconnected to camera!")
                else:
                    st.error(f"Disconnection failed: {response.text}")
            except Exception as e:
                st.error(f"Disconnection error: {str(e)}")

    # --- Configuration Section ---
    if st.session_state.connection_status == "connected":
        with st.expander("⚙️ Camera Configuration", expanded=True):
            st.session_state.config["integration_time_ms"] = st.number_input(
                "Integration Time (ms)",
                min_value=1,
                max_value=1000,
                value=st.session_state.config["integration_time_ms"],
            )

            st.session_state.config["cube_width"] = st.number_input(
                "Width (pixels)",
                min_value=100,
                max_value=4096,
                value=st.session_state.config["cube_width"],
            )

            st.session_state.config["cube_height"] = st.number_input(
                "Height (pixels)",
                min_value=100,
                max_value=4096,
                value=st.session_state.config["cube_height"],
            )

            st.session_state.config["preview"] = st.toggle("Preview")

            if st.button("Apply Configuration"):
                response = requests.post(
                    f"{st.session_state.full_url}/setconfig",
                    json=st.session_state.config,
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "configured"
                    st.success("Configuration applied successfully!")
                else:
                    st.error(f"Configuration failed: {response.text}")

    # --- Acquisition Workflow ---
    if (
        st.session_state.connection_status != "connected"
        and st.session_state.connection_status != "disconnected"
    ):
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Take Dark Reference"):
                response = requests.post(
                    f"{st.session_state.full_url}/dark_reference", json={}
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "dark_reference"
                    st.success("Dark reference captured!")

        with col2:
            if st.button(
                "Take White Reference",
                # disabled=st.session_state.connection_status != "dark_reference",
            ):
                response = requests.post(
                    f"{st.session_state.full_url}/white_reference", json={}
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "white_reference"
                    st.success("White reference captured!")

        with col3:
            if st.button(
                "Acquire Object",
                # disabled=st.session_state.connection_status != "white_reference",
            ):
                response = requests.post(
                    f"{st.session_state.full_url}/acquire_object", json={}
                )
                if response.status_code == 200:
                    st.session_state.connection_status = "acquired"
                    st.success("Object acquisition complete!")

    # --- Data Retrieval ---
    if st.session_state.connection_status == "acquired":
        st.markdown("---")
        st.subheader("Data Retrieval")

        col_prev, col_full = st.columns(2)
        with col_prev:
            if st.button("Get Latest Preview"):
                response = requests.get(
                    f"{st.session_state.full_url}/get_latest_preview"
                )
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    st.image(img, caption="Latest Preview", use_container_width=True)

        with col_full:
            if st.button("Get Full Capture"):
                response = requests.get(f"{st.session_state.full_url}/get_latest")
                if response.status_code == 200:
                    st.session_state.latest_tar_gz = response.content
                    st.success("tar.gz file received!")
                else:
                    st.error("Error getting tar.gz file")
