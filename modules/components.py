import streamlit as st

def show_help_page():
    st.header("❓ Help & Documentation")
    st.markdown("---")
    st.subheader("Welcome to Apteryx!")
    st.write("""
    Apteryx helps researchers assess kiwi fruit ripeness and firmness using hyperspectral imaging.
    This application provides an interface to control the camera, visualize data, and run predictions.

    **Navigation:**
    Use the buttons on the left sidebar to switch between screens:
    - **Camera Control:** Configure camera settings, capture hyperspectral images.
    - **Data Visualization:** Upload existing hyperspectral data (.hdr/.raw) to view spectral plots and features.
    - **Datacube Viewer:** Explore individual bands of a hyperspectral datacube.
    - **Inference Engine:** Upload hyperspectral data to get ripeness and firmness predictions from the ML model.

    **File Formats:**
    The system primarily uses ENVI hyperspectral file formats:
    - `.hdr`: Header file containing metadata about the image.
    - `.raw`/`.bin`: Binary file containing the raw pixel data.
    Ensure both files are provided together when uploading.

    **Camera Settings (Camera Control):**
    - **Integration Time:** Sensor exposure duration.
    - **Width/Height:** Image dimensions in pixels.
    - **Estimates:** Rough RAM/file size based on settings.

    **Data Visualization Options:**
    - **Average Spectrum:** Mean reflectance per wavelength.
    - *(Other options are placeholders).*

    **Inference:**
    - **Ripeness:** Categorical (Unripe, Ripe, Overripe).
    - **Firmness:** Numerical (0-10).
    - **Confidence:** Model's prediction confidence.

    For more details, refer to the project's main documentation (link would go here).
    """)

def show_feedback_page():
    st.header("📧 Feedback")
    st.markdown("---")
    st.write("We value your input! Please share any feedback, bug reports, or feature requests below.")

    with st.form("feedback_form"):
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Comment", "Question"])
        feedback_email = st.text_input("Your Email (Optional)")
        feedback_text = st.text_area("Your Feedback:", height=150, placeholder="Describe your feedback here...")
        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            if feedback_text:
                st.success("Thank you for your feedback! (Simulated submission)")
                print(f"Feedback Received:\nType: {feedback_type}\nEmail: {feedback_email}\nMessage: {feedback_text}")
            else:
                st.error("Please enter your feedback before submitting.") 