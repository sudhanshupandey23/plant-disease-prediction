import streamlit as st
import numpy as np
from PIL import Image

# Ensure TensorFlow is available
try:
    model = tf.keras.models.load_model("trained_model.h5")
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Model loading error: {e}")

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #f5f7fa, #c3cfe2); }
    .stButton>button { background: #6a11cb; color: white; border-radius: 10px; padding: 10px; }
    .stButton>button:hover { background: #2575fc; transform: scale(1.05); }
    .stFileUploader>div>div>div>div { color: #6a11cb; }
    .stMarkdown h1 { color: #6a11cb; text-align: center; font-size: 2.5rem; }
    .stMarkdown h2 { color: #2575fc; font-size: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 Plant Disease Recognition")
st.markdown("Upload an image of a plant leaf, and our model will analyze it to detect any signs of disease.")

# File Uploader
test_image = st.file_uploader("Upload an image of a plant leaf:", type=["jpg", "jpeg", "png"])

# Function for prediction
def model_prediction(image_file):
    try:
        image = Image.open(image_file)
        image = image.resize((128, 128))  # Resize for model
        input_arr = tf.keras.preprocessing.image.img_to_array(image) / 255.0  # Normalize
        input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension

        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return predicted class index

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# Display uploaded image
if test_image:
    st.image(test_image, caption="Uploaded Image", use_column_width=True)
    st.success("✅ Image uploaded successfully!")

    if st.button("🔍 Predict Disease"):
        with st.spinner("🔬 Analyzing... Please wait"):
            result_index = model_prediction(test_image)

            if result_index is not None:
                class_names = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
                    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight',
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy',
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 'Tomato___Target_Spot',
                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
                ]
                st.success(f"🎯 Prediction: **{class_names[result_index]}**")
                st.balloons()
            else:
                st.error("❌ Failed to make a prediction. Please try again.")
