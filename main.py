import time
import easyocr
from helpers import *
from config import config
from yolo5_predict import predict_logits, load_YOLOv5_model, filter_bounding_boxes, crop_title_boxes


def main():
    # Access concatenated paths
    images_path = config.get_original_dataset_path('images')
    annotations_path = config.get_original_dataset_path('labels')
    meta_path = config.get_original_dataset_path('meta')

    data_folder = config.get('dataset_folder')
    dataset_images = config.get_dataset_path('images')
    dataset_labels = config.get_dataset_path('labels')

    # print paths

    cat_pth(images_path, 'Original DS images')
    cat_pth(annotations_path, 'Original DS labels')
    cat_pth(data_folder, 'Original DS path')
    cat_pth(dataset_images, 'DS images')
    cat_pth(dataset_labels, 'DS labels')
    cat_pth(meta_path, 'meta.json path')


if __name__ == '__main__':
    # Get argument to show results or not
    show_results = True

    # Path to your custom weights file
    model = load_YOLOv5_model('exp8')

    # set model to inference mode
    if model.training:
        model.eval()

    # Loading easyOCR model in memory
    reader = easyocr.Reader(['ar'], gpu=False, quantize=True, detector=False, recognizer=True)

    # Image Path
    # img = 'tests/imgs/00190.png' # confidence metric
    img = '../OCR/tests/imgs/0032.png'

    start_time = time.time()

    # Inference
    logits = predict_logits(img, model)

    # Get bounding boxes
    output, bboxes, confidences, classes = bboxes_confidences_model_output(logits)

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Inference time: {elapsed_time} seconds")

    # Load image to crop Title regions
    image = load_img(img)

    # Filter the bounding boxes by Class title and sort from Top to Bottom
    sorted_title_boxes = filter_bounding_boxes(np.copy(image), bboxes, classes, confidences, show_results)

    if show_results:
        # Show results and crop images
        title_images = crop_title_boxes(image, sorted_title_boxes, show_results)

        # Display OCR
        processed_titles = [OCR_preprocess_image(title_img) for title_img in title_images]
    # preprocessed_image = OCR_preprocess_image(title_images[0])

    swap_columns(sorted_title_boxes, 1, 2)
    start = time.time()
    # [x_min, x_max, y_min, y_max] we need to swap two columns from the sorted_title_boxes

    result = reader.recognize(image, detail=0, paragraph=True, batch_size=1,
                              horizontal_list=sorted_title_boxes, free_list=[])
    end = time.time()
    print(result)
    print("OCR Time: ", end - start)
