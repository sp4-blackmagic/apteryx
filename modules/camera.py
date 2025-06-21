import streamlit as st
import requests
from PIL import Image
import numpy as np
import io
import httpx
import tarfile
import os
import asyncio


PREPROCESSOR_API_URL_DEFAULT = "http://0.0.0.0:8001/preprocessor/api/preprocess"
STORAGE_ENDPOINT_VALUE_DEFAULT = "http://0.0.0.0:8000/upload"


async def post_files_to_preprocessor(hdr_basename, hdr_data, raw_basename, raw_data):
    """
    Asynchronously posts HDR and RAW file data to the preprocessor service.
    """
    preprocessor_url = st.session_state.get(
        "preprocessor_url", PREPROCESSOR_API_URL_DEFAULT
    )
    storage_endpoint_val = st.session_state.get(
        "storage_endpoint_val", STORAGE_ENDPOINT_VALUE_DEFAULT
    )

    files_payload = {
        "hdr_file": (hdr_basename, io.BytesIO(hdr_data)),
        "cube_file": (raw_basename, io.BytesIO(raw_data)),
    }
    data_payload = {
        "remove_background": "false",
        "min_wavelength": "470",
        "extra_features": "false",
        "target_bands": "224",
        "sg_polyorder_deriv": "2",
        "resampling_kind": "linear",
        "storage_endpoint": storage_endpoint_val,
        "max_wavelength": "900",
        "multiple_samples": "false",
        "extraction_methods": "",
        "sg_window_deriv": "11",
        "background_treshold": "0.1",
    }

    # Use a longer timeout for potentially large file uploads
    # Increased timeout to 60 seconds, adjust as needed
    timeout_seconds = st.session_state.get("preprocessor_timeout", 6000.0)

    async with httpx.AsyncClient() as client:
        try:
            request = client.build_request(
                "POST",
                preprocessor_url,
                files=files_payload,
                data=data_payload,
            )
            print(f"Prepared request to preprocessor: {request.url}")
            print(f"Request headers: {request.headers}")
            print(f"Request content: {request.content}")
            st.info(f"Sending data to preprocessor at {preprocessor_url}...")
            response = await client.post(
                preprocessor_url,
                files=files_payload,
                data=data_payload,
                timeout=timeout_seconds,
            )
            if response.is_error:
                await response.aread()
                response.raise_for_status()
            # response = await client.get(preprocessor_url)
            st.info(
                f"Response from preprocessor: {response.status_code} - {response.text}"
            )
            return response
        except httpx.TimeoutException:
            st.error(
                f"Request to preprocessor timed out after {timeout_seconds} seconds."
            )
            return None
        except httpx.RequestError as exc:
            st.error(f"Request to preprocessor failed: {exc}")
            return None
        except Exception as e:
            st.error(
                f"An unexpected error occurred while sending data to preprocessor: {e}"
            )
            return None


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
            if st.button("Get Full Capture & Send to Preprocessor"):
                try:
                    get_latest_response = requests.get(
                        f"{st.session_state.full_url}/get_latest"
                    )
                    if get_latest_response.status_code == 200:
                        st.success(
                            "Successfully retrieved capture data (tar.gz) from camera."
                        )
                        tar_gz_content = get_latest_response.content

                        hdr_file_content = None
                        hdr_filename = None
                        raw_file_content = None
                        raw_filename = None

                        try:
                            with io.BytesIO(tar_gz_content) as bio:
                                with tarfile.open(fileobj=bio, mode="r:gz") as tar:
                                    for member in tar.getmembers():
                                        if member.isfile():
                                            base_name = os.path.basename(member.name)
                                            if (
                                                base_name.lower().endswith((".hdr"))
                                                and hdr_file_content is None
                                            ):
                                                hdr_file_content = tar.extractfile(
                                                    member
                                                ).read()
                                                hdr_filename = base_name
                                            # Common hyperspectral raw extensions: .raw, .img, .bil, .dat
                                            elif (
                                                base_name.lower().endswith(
                                                    (".raw", ".img", ".bil", ".dat")
                                                )
                                                and raw_file_content is None
                                            ):
                                                raw_file_content = tar.extractfile(
                                                    member
                                                ).read()
                                                raw_filename = base_name

                            if (
                                hdr_file_content
                                and raw_file_content
                                and hdr_filename
                                and raw_filename
                            ):
                                st.success(
                                    f"Extracted HDR ('{hdr_filename}') and RAW ('{raw_filename}') files from archive."
                                )

                                # Run the async function to post files
                                preprocessor_response = asyncio.run(
                                    post_files_to_preprocessor(
                                        hdr_filename,
                                        hdr_file_content,
                                        raw_filename,
                                        raw_file_content,
                                    )
                                )

                                if preprocessor_response:
                                    if (
                                        preprocessor_response.status_code == 200
                                        or preprocessor_response.status_code == 201
                                    ):  # Common success codes
                                        st.success(
                                            f"Data successfully sent to preprocessor: {preprocessor_response.status_code}"
                                        )
                                        try:
                                            st.json(
                                                preprocessor_response.json()
                                            )  # Display JSON response if any
                                        except:
                                            st.text(
                                                preprocessor_response.text
                                            )  # Display text response if not JSON
                                    else:
                                        st.error(
                                            f"Preprocessor returned an error: {preprocessor_response.status_code}"
                                        )
                                        st.text(
                                            preprocessor_response.text
                                        )  # Show error response text
                            else:
                                missing_files = []
                                if not hdr_file_content:
                                    missing_files.append("HDR")
                                if not raw_file_content:
                                    missing_files.append("RAW")
                                st.error(
                                    f"Could not find {' and '.join(missing_files)} file(s) in the tar.gz archive. Ensure files have standard extensions (.hdr, .raw, .img, .bil, .dat)."
                                )

                        except tarfile.TarError as te:
                            st.error(f"Error reading tar.gz archive: {te}")
                        except Exception as e_extract:
                            st.error(
                                f"An error occurred during file extraction or sending: {e_extract}"
                            )

                    else:
                        st.error(
                            f"Error getting tar.gz file from camera: {get_latest_response.status_code} - {get_latest_response.text}"
                        )
                except requests.exceptions.RequestException as e_req:
                    st.error(
                        f"Communication error with camera for Get Full Capture: {e_req}"
                    )
                except Exception as e_main:
                    st.error(
                        f"An unexpected error occurred in 'Get Full Capture': {e_main}"
                    )
