# Arabic-Book-Title-Recognition
This project provides an OCR (Optical Character Recognition) solution for detecting and recognizing Book titles from Images of Arabic Books. It utilizes YOLOv5 for text detection and EasyOCR for text recognition. The application is containerized using Docker and exposes API endpoints via Flask.

## Methodology 
- YOLOv5: Transfer Learning for 40 epochs for text detection on the Books dataset from [arabic-documents-ocr-dataset-kaggle](https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset/data).
- EasyOCR: For text recognition only.
- Flask: Python micro web framework for API endpoints.
- Docker: Containerization technology for easy deployment.

## Demo and Inference Results
![00231 (1)](https://github.com/user-attachments/assets/3825167d-366c-4822-8a44-c590e4b59b1c)

![val_batch2_pred](https://github.com/user-attachments/assets/9de7ae05-3ae6-4f17-8b4f-4b5b50ed3328)

## Yolov5 Training Results and Metircs
Title Object Detection achieves 66 mAp.
![results](https://github.com/user-attachments/assets/13c251d9-d65e-499e-8a82-a73a945da34e)

### PR and F1
![PR_curve](https://github.com/user-attachments/assets/2e58b3fa-1b5f-411a-9e72-77a571809195)
![F1_curve](https://github.com/user-attachments/assets/03cfd904-a6c1-4f1e-82d3-4223f2cef9e0)

### Confusion Matrix (Only interested in Titles)
![confusion_matrix](https://github.com/user-attachments/assets/39e5b2e1-7e20-4cfa-9a85-ab2709f4f66f)

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
Value: Select your image file to upload.

### Response
Returns Title string in arabic.

## Resources
- YOLOv5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- EasyOCR: [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
- Dataset: [humansintheloop/arabic-documents-ocr-dataset-kaggle](https://www.kaggle.com/datasets/humansintheloop/arabic-documents-ocr-dataset/data)
