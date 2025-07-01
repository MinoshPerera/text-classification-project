from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', prediction="No image uploaded")

        image = request.files['image']
        filename = secure_filename(image.filename)
        image_path = os.path.join('uploads', filename)
        image.save(image_path)

        predicted_class = predict_image(image_path)
        return render_template('index.html', prediction=predicted_class)

    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    filename = secure_filename(image.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(image_path)

    predicted_class = predict_image(image_path)
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)