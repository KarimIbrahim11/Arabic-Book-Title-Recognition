from PIL import Image
from config import config
from flask import Flask, request, jsonify
from model import load_model, process_image
import os
import io

app = Flask(__name__)

# Load the model once when the server starts
yoloV5, ocr = load_model(exp='exp11')

config.config['UPLOAD_FOLDER'] = 'uploads'
config.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return "Arabic OCR Flask is running"


@app.route('/predict', methods=['POST'])
def predict():
    print("Request: ", request)
    print("File received:", request.files['file'])
    print("Config:", config.get('uploads', 'imgs'))

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 404

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # if file and allowed_file(file.filename):
    if file:
        filepath = 'uploads/imgs'+str(file.filename)
        file.save(filepath)

        try:
            sorted_bboxes, titles, objectness = process_image(file, yoloV5, ocr, False)

            return jsonify({
                'bounding_boxes': sorted_bboxes,
                'title': ' '.join([title for title in titles])
            })
        except FileNotFoundError as e:
            print("Exception raised", e)
            return jsonify({
                'error': "File not found was raised" + e.__str__()
            }), 400
        except Exception as e:
            print("Exception raised", e)
            return jsonify({
                'error': "Unexcpected error arised" + e.__str__()
            }), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
