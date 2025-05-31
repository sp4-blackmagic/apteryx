import streamlit as st

def show_about_page():
    st.header("👥 About Us")
    st.markdown("---")

    # Team Introduction
    st.markdown("""
    ## Our Team
    We are a group of passionate students from the University of Southern Denmark, working together to develop innovative solutions in hyperspectral imaging and machine learning.
    """)

    # Team Members
    team_members = [
        {
            "name": "Bence Kanyok",
            "email": "bekan23@student.sdu.dk",
            "role": "Team Member"
        },
        {
            "name": "Daniel Gardev",
            "email": "dagar23@student.sdu.dk",
            "role": "Team Member"
        },
        {
            "name": "Francesco Schenone",
            "email": "fsche23@student.sdu.dk",
            "role": "Team Member"
        },
        {
            "name": "Jan Pawel Musiol",
            "email": "jamus23@student.sdu.dk",
            "role": "Team Member"
        },
        {
            "name": "Kacper Grzegorz Grzyb",
            "email": "kagrz23@student.sdu.dk",
            "role": "Team Member"
        },
        {
            "name": "Phongsakon Mark Konrad",
            "email": "phkon23@student.sdu.dk",
            "role": "Team Member"
        }
    ]

    # Create columns for team members
    cols = st.columns(3)
    for idx, member in enumerate(team_members):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h3 style='color: #1f77b4; margin-bottom: 10px;'>{member['name']}</h3>
                <p style='color: #666; margin-bottom: 5px;'>{member['role']}</p>
                <p style='color: #666;'>{member['email']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Project Description
    st.markdown("""
    ## About Apteryx
    Apteryx is a cutting-edge application that leverages hyperspectral imaging technology to assess kiwi fruit quality. 
    Our project combines advanced machine learning techniques with spectral analysis to provide accurate predictions of 
    ripeness and firmness, revolutionizing the way we evaluate fruit quality.

    ### Our Mission
    To develop innovative solutions that bridge the gap between technology and agriculture, making quality assessment 
    more efficient and accurate for farmers and distributors.

    ### Technology Stack
    - Hyperspectral Imaging
    - Machine Learning
    - Computer Vision
    - Data Analysis
    - Web Development
    """)

    # Contact Information
    st.markdown("""
    ## Contact
    For any inquiries or collaboration opportunities, please feel free to reach out to our team members.
    """) 