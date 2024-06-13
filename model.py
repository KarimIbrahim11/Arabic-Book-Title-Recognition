import torch
import numpy as np
import easyocr
from yolo5_predict import load_YOLOv5_model, predict_logits, filter_bounding_boxes, crop_title_boxes
from helpers import *

def load_model(exp='exp8', gpu=False):
    # Load your model here
    # Path to your custom weights file
    model = load_YOLOv5_model(exp)

    # set model to inference mode
    if model.training:
        model.eval()

    # Loading easyOCR model in memory
    reader = easyocr.Reader(['ar'], gpu=gpu, quantize=True, detector=False, recognizer=True)
    return model, reader

def process_image(file, model, reader, show_results=False):
    # Transform the PIL image to a tensor
    img = Image.open(file)
    img_path = 'uploads/imgs/' + file.filename
    img.save(img_path)

    file = img_path
    logits = predict_logits(file, model)

    img = load_img(file)

    # Get bounding boxes
    _, bboxes, confidences, classes = bboxes_confidences_model_output(logits)

    # Filter the bounding boxes by Class title and sort from Top to Bottom
    sorted_title_boxes = filter_bounding_boxes(np.copy(img), bboxes, classes, confidences, show_results)

    if show_results:
        # Show results and crop images
        _ = crop_title_boxes(img, sorted_title_boxes, show_results)

    swap_columns(sorted_title_boxes, 1, 2)
    # [x_min, x_max, y_min, y_max] we need to swap two columns from the sorted_title_boxes

    result = reader.recognize(img, detail=0, paragraph=True, batch_size=1,
                              horizontal_list=sorted_title_boxes, free_list=[])

    return sorted_title_boxes, result, confidences
