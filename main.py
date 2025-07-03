import streamlit as st

import numpy as np

# ------------------------
# Load Model (Cached)
# ------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('trained_model.keras')
    return model

model = load_model()  # Load once

# ------------------------
# Model Prediction Function
# ------------------------
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Make it a batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# ------------------------
# Sidebar Navigation
# ------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ------------------------
# Home Page
# ------------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant.
    2. **Analysis:** The system uses deep learning to analyze the leaf.
    3. **Results:** See the disease prediction instantly.

    ### Why Choose Us?
    - 🔍 **Accurate Predictions**
    - ⚡ **Fast Processing**
    - 🧑‍🌾 **Farmer Friendly**

    ### Get Started
    Click on the **Disease Recognition** page to begin.
    """)

# ------------------------
# About Page
# ------------------------
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### 📊 Dataset Information
    This dataset consists of about 87,000 RGB images of healthy and diseased crop leaves across 38 categories.
    
    - **Train:** 70,295 images  
    - **Validation:** 17,572 images  
    - **Test:** 33 images (custom prediction)

    #### 📦 Source
    Recreated using offline augmentation from the original dataset [GitHub repo](https://github.com/spMohanty/PlantVillage-Dataset).

    #### 🧠 Model
    - CNN-based architecture
    - Trained using TensorFlow/Keras
    - Input size: 128x128 RGB
    """)

# ------------------------
# Prediction Page
# ------------------------
elif app_mode == "Disease Recognition":
    st.header("🧪 Disease Recognition")

    test_image = st.file_uploader("📁 Choose a plant leaf image:")

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        if test_image is not None:
            with st.spinner("⏳ Predicting... Please wait."):
                result_index = model_prediction(test_image)

                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]

                st.success(f"✅ Prediction: **{class_name[result_index]}**")
        else:
            st.warning("⚠️ Please upload an image first.")
