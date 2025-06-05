import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load model when app starts
model = None

def load_model():
    global model
    # Load the .h5 model
    model_path = os.path.join('model', 'model.h5')
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")

def preprocess_image(image_bytes):
    """Preprocess the image for model prediction"""
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize image if needed (adjust as per your model requirements)
    image = image.resize((224, 224))
    
    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension if needed
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'message': 'TensorFlow model is running. Send POST requests to /predict.'
    })

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the model is loaded
        if model is None:
            load_model()
        
        try:
            # Get base64 encoded image from the request
            json_data = request.json
            if 'image' not in json_data:
                return jsonify({'error': 'No image provided'}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(json_data['image'])
            
            # Preprocess the image
            processed_image = preprocess_image(image_data)
            
            # Make prediction
            prediction = model.predict(processed_image)
            
            # Process the prediction (adjust as needed for your model)
            result = {
                'prediction': prediction.tolist(),
                # Add any additional processing here
            }
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    # Load model at startup
    load_model()
    
    # Get port from environment variable (for Railway)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)