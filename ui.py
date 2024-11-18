import streamlit as st
from Mushroom_Classifier import MushroomClassifier

# Streamlit page configuration
st.set_page_config(page_title='Mushroom Classifier', layout='wide', initial_sidebar_state='expanded')

# Custom CSS styling
st.markdown("""
    <style>
        body {
            font-family: 'Roboto', sans-serif;
            background-color: #f4f4f9;
            color: #333;
        }

        .heading-title {
            font-size: 26px;
            font-weight: 700;
            color: #FF6347;  
            text-align: center;
            margin-bottom: 25px;
            text-transform: uppercase;
            letter-spacing: 1px;  
            font-family: 'Montserrat', sans-serif; 
            border-bottom: 2px solid #4CAF50;  
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar with emoji and heading
st.sidebar.markdown(
    """
    <div style="margin-bottom: 20px;">
        <img src="https://em-content.zobj.net/source/twitter/348/mushroom_1f344.png" width="180" height="180" />
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown('<div class="heading-title">Classify</div>', unsafe_allow_html=True)

# Instructions section
st.sidebar.markdown("""
    <div style="font-size: 16px;">
        <h3>Instructions:</h3>
        <ul>
            <li>Ensure the mushroom image is clear, well-lit, and focused for accurate classification.</li>
            <li>Upload the image, and the app will classify it and provide safety information.</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# File uploader 
uploaded_file = st.sidebar.file_uploader("Choose a mushroom image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    classifier = MushroomClassifier()
    response = classifier.classify_image("temp_image.jpg")
    st.image(uploaded_file, caption='Uploaded Mushroom Image', use_column_width=True)
    st.markdown(response.content)
