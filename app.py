from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load your model (adjust the path as needed)
MODEL_PATH = 'models/wound_model.h5'
model = load_model(MODEL_PATH)

# Define class names
CLASSES = ['Bakar', 'Laserasi', 'Lecet']

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    img_bytes = file.read()
    img_array = preprocess_image(img_bytes)
    preds = model.predict(img_array)[0]
    max_idx = np.argmax(preds)
    result = {
        'class': CLASSES[max_idx],
        'confidence': float(preds[max_idx])
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)