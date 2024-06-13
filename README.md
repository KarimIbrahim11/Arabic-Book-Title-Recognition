# Arabic-Book-Title-Recognition
This project provides an OCR (Optical Character Recognition) solution for detecting and recognizing book titles in Arabic. It utilizes YOLOv5 for text detection and EasyOCR for text recognition. The application is containerized using Docker and exposes API endpoints via Flask.

## Installation
To run the Book Title OCR application, Docker must be installed on your system.
## Clone the Repository
```
git clone https://github.com/your-repo/book-ocr-docker.git
cd book-ocr-docker
```
## Build the Image
```
docker build -t book-ocr-docker .
```
## Usage
Run the Docker Container
```
docker run -p 5000:5000 book-ocr-docker
```
The application will be accessible at http://127.0.0.1:5000/.

# API Endpoints
The following API endpoints are available:

## Endpoint for Uploading an Image
URL: http://127.0.0.1:5000/predict
Method: [POST]
Description: Upload an image containing Arabic Book for OCR processing.

### Request Body
form-data
Key: image in format {.png, .jpg, .jpeg}

## Methodologies Used
YOLOv5: Trained it for 40 epochs for text detection.
EasyOCR: For text recognition.
Flask: Python micro web framework for API endpoints.
Docker: Containerization technology for easy deployment.

## Credits
YOLOv5: https://github.com/ultralytics/yolov5
EasyOCR: https://github.com/JaidedAI/EasyOCR

Value: Select your image file to upload.
