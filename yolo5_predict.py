import torch
import cv2
from helpers import *
from config import config


def load_YOLOv5_model(experiment='exp8'):
    # Path to your custom weights file
    model_pth = config.get('model_path')
    custom_weights = config.get_model_weights(experiment)

    # Load the model with your custom weights
    model = torch.hub.load(model_pth, 'custom', path=custom_weights, source='local')

    return model


def predict_logits(img, model):
    return model(img)


# # Display the results
# results.show()

def filter_bounding_boxes(image, bboxes, classes, confidences, show_results=False):
    if show_results:
        image_window_name = 'Image with Bounding Boxes'
        cv2.namedWindow(image_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(image_window_name, 800, 600)  # Width: 800, Height: 600

    height, width, _ = image.shape
    bbox_points = []

    # if not titles found, get the max confidence text
    if 0 not in classes:
        classes[np.argmax(confidences)] = 0

    for bbox in bboxes[classes == 0]:
        # Filter only labels with class 0 aka Title
        x1, y1, x2, y2 = bbox_retrieval(bbox, width, height)
        bbox_points.append([x1, y1, x2, y2])
        if show_results:
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=3)

    sorted_bounding_boxes = sorted(bbox_points, key=lambda bbox: bbox[1])
    if show_results:
        cv2.imshow(image_window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return sorted_bounding_boxes


def crop_title_boxes(image, sorted_bounding_boxes, show_results=False):
    if show_results:
        title_window_name = 'Title'
        cv2.namedWindow(title_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(title_window_name, 800, 600)  # Width: 800, Height: 600
    title_images = []
    for bbox in sorted_bounding_boxes:
        title_image = crop_text_regions(tuple(bbox), image)
        title_images.append(title_image)
        if show_results:
            cv2.imshow(title_window_name, title_image)
            cv2.waitKey(0)
    if show_results:
        cv2.destroyAllWindows()

    return title_images
