import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Fashion Item Classifier",
    page_icon="👗",
    layout="centered",
    initial_sidebar_state="auto"
)

# Class labels
class_names = ['Shirt', 'T-Shirt', 'Hoodies', 'Jeans', 'Shorts', 'Kurtas', 'Blazers']

@st.cache_resource
def load_model():
    """Load the TensorFlow model"""
    try:
        model = tf.keras.models.load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Resize image to 224x224
    img = image.resize((224, 224))
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def make_prediction(model, image):
    """Make prediction on image"""
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Create probabilities dictionary
        all_probabilities = {
            class_name: float(prob) 
            for class_name, prob in zip(class_names, predictions[0])
        }
        
        return predicted_class, confidence, all_probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        st.error(f"Error decoding base64 image: {str(e)}")
        return None

def main():
    # Header
    st.title("👗 Fashion Item Classifier")
    st.markdown("**AI-powered fashion item classification**")
    st.markdown("Classify your fashion items into: Shirt, T-Shirt, Hoodies, Jeans, Shorts, Kurtas, Blazers")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model could not be loaded. Please ensure 'Model2.h5' is in the project directory.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for input method selection
    st.sidebar.header("📤 Input Method")
    input_method = st.sidebar.selectbox(
        "Choose how to input your image:",
        ["Upload Image File", "Base64 Input", "Camera Capture"]
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Input Image")
        
        image = None
        
        if input_method == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                help="Upload an image of a fashion item"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        elif input_method == "Base64 Input":
            st.markdown("**Paste your base64 encoded image:**")
            base64_input = st.text_area(
                "Base64 Image Data",
                height=150,
                placeholder="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA...",
                help="Paste the complete base64 string including data URL prefix"
            )
            
            if base64_input.strip():
                image = base64_to_image(base64_input.strip())
                if image:
                    st.image(image, caption="Base64 Image", use_column_width=True)
        
        elif input_method == "Camera Capture":
            camera_image = st.camera_input("Take a photo of your fashion item")
            
            if camera_image is not None:
                image = Image.open(camera_image)
                st.image(image, caption="Camera Image", use_column_width=True)
    
    with col2:
        st.header("🎯 Prediction Results")
        
        if image is not None:
            if st.button("🔍 Classify Fashion Item", type="primary", use_container_width=True):
                with st.spinner("Analyzing your fashion item..."):
                    predicted_class, confidence, all_probabilities = make_prediction(model, image)
                    
                    if predicted_class is not None:
                        # Main prediction result
                        st.success("🎉 Classification Complete!")
                        
                        # Display main result
                        st.metric(
                            label="Predicted Item",
                            value=predicted_class,
                            delta=f"{confidence:.1%} confidence"
                        )
                        
                        # Confidence bar
                        st.progress(confidence)
                        
                        # All probabilities
                        st.subheader("📊 All Probabilities")
                        
                        # Sort probabilities in descending order
                        sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
                        
                        for class_name, prob in sorted_probs:
                            # Create a colored bar based on probability
                            if class_name == predicted_class:
                                st.markdown(f"**{class_name}**: {prob:.3f} ({prob:.1%})")
                                st.progress(prob)
                            else:
                                st.markdown(f"{class_name}: {prob:.3f} ({prob:.1%})")
                                st.progress(prob)
                        
                        # Additional info
                        st.info(f"💡 The model is {confidence:.1%} confident that this is a **{predicted_class}**")
        else:
            st.info("👆 Please upload an image, provide base64 data, or take a photo to get started!")
    
    # Footer with model info
    st.markdown("---")
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Details:**
        - Input Size: 224x224 pixels
        - Classes: 7 fashion categories
        - Architecture: Convolutional Neural Network
        - Image Format: RGB images, normalized to [0,1]
        
        **Supported Categories:**
        - Shirt
        - T-Shirt
        - Hoodies
        - Jeans
        - Shorts
        - Kurtas
        - Blazers
        """)
    
    # API equivalent section
    with st.expander("🔧 API Equivalent"):
        st.markdown("""
        **This Streamlit app provides the same functionality as these API endpoints:**
        
        ```bash
        # File upload equivalent
        POST /predict
        
        # Base64 input equivalent  
        POST /predict/base64
        ```
        
        **Sample API Response Format:**
        ```json
        {
          "predicted_class": "T-Shirt",
          "confidence": 0.95,
          "all_probabilities": {
            "Shirt": 0.02,
            "T-Shirt": 0.95,
            "Hoodies": 0.01,
            "Jeans": 0.01,
            "Shorts": 0.00,
            "Kurtas": 0.01,
            "Blazers": 0.00
          }
        }
        ```
        """)

if __name__ == "__main__":
    main()
