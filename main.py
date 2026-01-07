"""
🎬 MELORA - Film Recommendation System
Landing Page - No Scroll (Logo Safe)
"""

import streamlit as st
import os, base64

# Page configuration
st.set_page_config(
    page_title="MELORA - Film Recommendation System",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    [data-testid="stSidebar"] { display: none; }

    .main {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        min-height: 100vh;  
        padding-top: 1.5rem; 
    }
            

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 0.8rem;
        max-width: 850px;
    }

    /* Subtitle / Page title */
    .hero-subtitle {
        font-size: 1.15rem;
        color: #000000;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }

    /* SMALLER CARD */
    .feature-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 9px;
        padding: 1rem 1.1rem;
        text-align: center;
    }

    .feature-icon {
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
    }

    .feature-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 0.4rem;
    }

    .feature-description {
        font-size: 0.85rem;
        color: #CBD5E1;
        line-height: 1.45;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: #000000;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.55rem 2rem;
        border-radius: 999px;
        border: none;
        margin-top: 0.9rem;
        width: 100%;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 0.72rem;
        margin-top: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Display logo as circle (SAFE VERSION - NOT CUT)
logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.jpeg')
if os.path.exists(logo_path):
    import base64

    with open(logo_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode()

    st.markdown(f"""
    <div style="
        display: flex;
        justify-content: center;
        margin-top: 24px;
        margin-bottom: 1.2rem;
    ">
        <img src="data:image/jpeg;base64,{img_base64}"
             style="
                width: 170px;
                height: 170px;
                border-radius: 50%;
                object-fit: cover;
                box-shadow: 0 10px 40px rgba(99, 102, 241, 0.3);
             ">
    </div>
    """, unsafe_allow_html=True)



# Page title (BLACK)
st.markdown("""
<div class="hero-subtitle">
    Film Recommendation System
</div>
""", unsafe_allow_html=True)

# Smaller Film Card
st.markdown("""
<div class="feature-card">
    <div class="feature-icon">🎬</div>
    <div class="feature-title">Explore Film</div>
    <div class="feature-description">
        Content-Based Filtering<br>
        7.400+ film data<br>
        Rekomendasi platform streaming
    </div>
</div>
""", unsafe_allow_html=True)

# Button
if st.button("Explore Film 🎥", use_container_width=True):
    st.switch_page("pages/2_Film.py")

# Footer
st.markdown("""
<div class="footer">
    MELORA • Final Project Kelompok 4
</div>
""", unsafe_allow_html=True)
