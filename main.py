import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Custom CSS for a modern and sleek design
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6a11cb, #2575fc);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2575fc, #6a11cb);
        transform: scale(1.05);
    }
    .stFileUploader>div>div>div>div {
        color: #6a11cb;
    }
    .stMarkdown h1 {
        color: #6a11cb;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    .stMarkdown h2 {
        color: #2575fc;
        font-size: 2rem;
        margin-bottom: 15px;
    }
    .stMarkdown h3 {
        color: #148F77;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    .stSuccess {
        color: #28B463;
        font-weight: bold;
    }
    .stError {
        color: #C0392B;
        font-weight: bold;
    }
    .stSpinner>div>div {
        border-color: #6a11cb transparent transparent transparent;
    }
    .stImage>img {
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Tensorflow Model Prediction
def model_prediction(test_image):
    """
    Predict the plant disease using the trained model.
    """
    try:
        model = tf.keras.models.load_model("trained_model.h5")
        image = Image.open(test_image)
        image = image.resize((128, 128))  # Resize the image to match model input size
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Sidebar
st.sidebar.title("🌱 Plant Disease Recognition")
st.sidebar.markdown("""
    **Navigate through the app using the options below:**
""")
app_mode = st.sidebar.radio(
    "Select Page",
    ["Home", "About", "Disease Recognition"],
    index=0
)

# Main Page
if app_mode == "Home":
    st.title("🌿 Welcome to the Plant Disease Recognition System!")
    st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 1.2rem; color: #555;">
            Our mission is to help you identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.image("home_page.jpeg", use_column_width=True, caption="Healthy Plants, Healthy Future")

    st.markdown("""
    ### 🚀 How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced machine learning algorithms.
    3. **Results:** View the results and recommendations for further action.

    ### 🌟 Why Choose Us?
    - **Accuracy:** State-of-the-art machine learning models for precise disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Get results in seconds.

    ### 🛠️ Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our system!
    """)

# About Project
elif app_mode == "About":
    st.title("📚 About")
    st.markdown("""
    <div style="text-align: justify;">
        This project aims to help farmers and gardeners identify plant diseases early and accurately. By leveraging machine learning, we provide a tool that can analyze images of plant leaves and detect diseases with high precision.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    #### 📂 About the Dataset
    - The dataset consists of **87,000 RGB images** of healthy and diseased crop leaves.
    - It is categorized into **38 different classes**.
    - The dataset is divided into:
        - **Training Set:** 70,295 images
        - **Validation Set:** 17,572 images
        - **Test Set:** 33 images
    """)

    st.markdown("""
    #### 🛠️ Technologies Used
    - **TensorFlow:** For building and training the deep learning model.
    - **Streamlit:** For creating this interactive web application.
    - **Python:** For backend logic and data processing.
    """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.title("🔍 Disease Recognition")
    st.markdown("""
    <div style="text-align: justify;">
        Upload an image of a plant leaf, and our system will analyze it to detect any signs of diseases. Ensure the image is clear and focused on the leaf for the best results.
    </div>
    """, unsafe_allow_html=True)

    # File Uploader
    test_image = st.file_uploader("Upload an image of a plant leaf:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Image uploaded successfully!")

        # Predict button
        if st.button("Predict Disease"):
            with st.spinner("🔬 Analyzing the image. Please wait..."):
                result_index = model_prediction(test_image)
                if result_index is not None:
                    st.balloons()
                    st.write("## 🎯 Prediction Result")
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    st.success(f"**Prediction:** The model predicts that the plant is affected by **{class_name[result_index]}**.")
                else:
                    st.error("❌ Failed to make a prediction. Please try again.")