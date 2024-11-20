import io
import numpy as np
import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
from flask_cors import CORS  # Correct import for CORS
import json

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for only certain origins (Make sure to use proper Python syntax)
CORS(app, origins=["http://localhost:3000"])  # Allow only localhost:3000 for frontend requests

# Load the model (ensure your .h5 model is in the same directory or provide the path)
model = load_model('plant_disease_prediction_model.h5')

# Load the class indices from the saved JSON file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices to create a mapping from index to class name
reverse_class_indices = {v: k for k, v in class_indices.items()}

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert RGBA images to RGB (removes the alpha channel)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize to model's expected input size (224x224 is common, but verify with your model)
    image = image.resize((224, 224))  # Adjust to your model's input size
    img_array = np.array(image)  # Convert image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (for Keras)
    img_array = img_array.astype("float32") / 255.0  # Normalize if required by your model
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Check if the image data exists in the request
        if 'image' not in data:
            return jsonify({'error': 'No image found in the request'}), 400

        image_data = data['image']
        
        # Remove data URL prefix if it exists
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]

        # Decode the base64 string
        image_bytes = base64.b64decode(image_data)

        # Preprocess the image
        image = preprocess_image(image_bytes)

        # Debugging: Show the shape of the input image
        print("Processed image shape:", image.shape)

        # Make a prediction
        prediction = model.predict(image)

        # Debugging: Show the raw prediction values (probabilities)
        print("Model prediction (raw output):", prediction)

        predicted_class_index = np.argmax(prediction, axis=1)[0]

        # Debugging: Check predicted class index
        print("Predicted class index:", predicted_class_index)

        # Get the name of the predicted class from the reverse mapping
        predicted_name = reverse_class_indices.get(predicted_class_index, "Unknown")

        # Return the prediction
        return jsonify({
            'prediction': predicted_name
        })

    except Exception as e:
        print(str(e))  # Log the error for debugging
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
