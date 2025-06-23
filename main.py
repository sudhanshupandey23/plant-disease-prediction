import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import traceback

# Page Config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱", layout="centered")

# Custom CSS for design
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
    .stMarkdown h1 {
        color: #6a11cb;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Cache the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_model.h5")

model = load_model()

# Prediction function
def model_prediction(test_image):
    try:
        image = Image.open(test_image)
        image = image.resize((128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = input_arr / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        predicted_index = np.argmax(predictions)
        confidence = float(np.max(predictions)) * 100
        return predicted_index, confidence
    except Exception as e:
        traceback.print_exc()
        st.error(f"❌ Error during prediction: {e}")
        return None, None

# Class labels
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

# Sidebar
st.sidebar.title("🌱 Plant Disease Recognition")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Recognition"])

# Home Page
if app_mode == "Home":
    st.title("🌿 Welcome to the Plant Disease Recognition System!")
    st.markdown("""
    <div style="text-align: center;">
        <p style="font-size: 1.2rem; color: #555;">
            Upload a plant image and let our AI tell you if it's healthy or diseased.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True, caption="Healthy Plants, Healthy Future")
    st.markdown("""
    ### 🚀 How It Works
    1. Go to **Disease Recognition** page.
    2. Upload an image of the leaf.
    3. Our model will analyze and predict the disease.
    
    ### 🌟 Features
    - High accuracy with deep learning
    - Simple and fast
    """)

# About Page
elif app_mode == "About":
    st.title("📚 About")
    st.markdown("""
    <div style="text-align: justify;">
        This tool uses a deep learning model to classify 38 plant disease categories from leaf images.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    #### 📂 Dataset Info
    - 87,000 RGB images
    - 38 categories
    - Used TensorFlow + CNN
    
    #### 🛠️ Tech Stack
    - **TensorFlow** (Model training)
    - **Streamlit** (Web app)
    - **Python & PIL** (Image processing)
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("🔍 Disease Recognition")
    st.markdown("""
    Upload a clear leaf image and let our model detect any disease.
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("📤 Upload a leaf image (jpg/png):", type=["jpg", "jpeg", "png"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        st.success("✅ Image uploaded successfully.")

        if st.button("🔬 Predict Disease"):
            with st.spinner("Analyzing... Please wait..."):
                result_index, confidence = model_prediction(test_image)
                if result_index is not None:
                    prediction = class_name[result_index]
                    st.balloons()
                    st.write("## 🎯 Prediction Result")
                    st.success(f"**Prediction:** {prediction}")
                    st.write(f"🔎 **Confidence:** {confidence:.2f}%")

                    if "healthy" in prediction.lower():
                        st.info("🌿 The plant looks healthy!")
                    else:
                        st.warning("⚠️ Disease detected. Please consult a specialist.")
                else:
                    st.error("❌ Could not predict. Please try again.")

# Footer
st.markdown("""
<hr>
<div style="text-align:center">
    Made with ❤️ by Sudhanshu Pandey | Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
