import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Custom CSS for UI enhancements
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
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2575fc;
        text-align: center;
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

# Function to Predict Plant Disease
def model_prediction(test_image):
    """
    Predicts the disease of the given plant image using the trained CNN model.
    """
    try:
        # Load trained model
        model = tf.keras.models.load_model("trained_model.h5")

        # Open and preprocess image
        image_data = Image.open(test_image)
        image_data = image_data.resize((128, 128))  # Resize image
        input_arr = np.array(image_data) / 255.0  # Normalize pixel values
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        # Debugging: Check the shape of the image before passing to model
        print("Image shape before prediction:", input_arr.shape)

        # Make prediction
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Get the predicted class index
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        return None

# Sidebar
st.sidebar.title("🌱 Plant Disease Recognition")
app_mode = st.sidebar.radio("Select Page", ["Home", "About", "Disease Recognition"], index=0)

# Home Page
if app_mode == "Home":
    st.title("🌿 Welcome to the Plant Disease Recognition System!")
    st.image("home_page.jpeg", use_column_width=True, caption="Healthy Plants, Healthy Future")

    st.markdown("""
    ### 🚀 How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page.
    2. **Analyze:** AI-based system processes the image.
    3. **Get Results:** Disease prediction and recommendations.

    ### 🌟 Why Choose Us?
    - **High Accuracy:** State-of-the-art machine learning model.
    - **User-Friendly:** Simple and intuitive interface.
    - **Fast & Efficient:** Get results in seconds.

    🔍 Click on **Disease Recognition** in the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.title("📚 About This Project")
    st.markdown("""
    - Uses **Machine Learning & Deep Learning** to detect plant diseases.
    - Dataset contains **87,000 images** with 38 plant disease classes.
    - Trained using **TensorFlow, Keras, and CNN models**.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.title("🔍 Plant Disease Detection")
    st.markdown("Upload an image of a plant leaf, and our AI system will analyze it for diseases.")

    # File Uploader
    test_image = st.file_uploader("Upload an image of a plant leaf:", type=["jpg", "jpeg", "png"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)
        st.success("✅ Image uploaded successfully!")

        # Predict button
        if st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing the image... Please wait ⏳"):
                result_index = model_prediction(test_image)
                
                if result_index is not None:
                    st.balloons()
                    st.write("## 🎯 Prediction Result")
                    
                    # Class Labels (Ensure they match the model output)
                    class_labels = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 
                        'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight', 'Grape___healthy', 
                        'Orange___Citrus_greening', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper___Bacterial_spot', 
                        'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
                        'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
                        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot', 
                        'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___mosaic_virus', 'Tomato___healthy'
                    ]
                    
                    # Display Prediction
                    st.success(f"**Prediction:** The plant is affected by **{class_labels[result_index]}**.")

                else:
                    st.error("❌ Failed to make a prediction. Please try again.")
