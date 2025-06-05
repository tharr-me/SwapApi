import streamlit as st
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io
import cv2

# Set page configuration
st.set_page_config(
    page_title="Image Classification App",
    page_icon="🖼️",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load the trained model"""
    model = tf.keras.models.load_model('model/model.h5')
    return model

# Load the model
model = load_model()

# Get class names (modify this according to your model)
class_names = ['class1', 'class2', 'class3']  # Replace with your classes

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    # Resize image
    img = image.resize(target_size)
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def decode_base64_image(base64_string):
    """Decode base64 string to image"""
    # Remove header if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    # Decode base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img

# Title and description
st.title("TensorFlow Image Classification")
st.markdown("Upload an image or provide a Base64 encoded image to classify")

# Create two tabs for different input methods
tab1, tab2 = st.tabs(["File Upload", "Base64 Input"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction when button is clicked
        if st.button("Classify Image", key="classify_upload"):
            with st.spinner("Classifying..."):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Make prediction
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class])
                
                # Display results
                st.success(f"Prediction: {class_names[predicted_class]}")
                st.progress(confidence)
                st.write(f"Confidence: {confidence:.2%}")

with tab2:
    # Base64 input
    base64_input = st.text_area("Paste Base64 encoded image:", height=150)
    
    if base64_input:
        try:
            # Decode base64 string to image
            image = decode_base64_image(base64_input)
            
            # Display the decoded image
            st.image(image, caption="Decoded Image", use_column_width=True)
            
            # Make prediction when button is clicked
            if st.button("Classify Image", key="classify_base64"):
                with st.spinner("Classifying..."):
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Make prediction
                    predictions = model.predict(processed_image)
                    predicted_class = np.argmax(predictions[0])
                    confidence = float(predictions[0][predicted_class])
                    
                    # Display results
                    st.success(f"Prediction: {class_names[predicted_class]}")
                    st.progress(confidence)
                    st.write(f"Confidence: {confidence:.2%}")
        
        except Exception as e:
            st.error(f"Error decoding base64 image: {e}")

# Add some additional information
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a TensorFlow model to classify images. "
    "You can upload an image file or paste a Base64 encoded image string."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Created by: tharr-mei")