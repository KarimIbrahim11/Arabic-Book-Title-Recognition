import os
import cv2
import json
import shutil
from PIL import Image, ImageDraw
from .file_utils import create_dir
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def bbox_retrieval(bbox, W_img, H_img):
    x, y, w, h = bbox
    x1 = (x - w / 2) * W_img
    y1 = (y - h / 2) * H_img
    x2 = (x + w / 2) * W_img
    y2 = (y + h / 2) * H_img
    return int(x1), int(y1), int(x2), int(y2)


def crop_text_regions(bbox, image):
    x1, y1, x2, y2 = bbox
    # cropped_image = image.crop(bbox)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def bboxes_confidences_model_output(results):
    output = results.xywhn[0].cpu().numpy()
    bboxes = output[:, :4]
    confidences = output[:, 4]
    classes = output[:, 5]
    return output, bboxes, confidences, classes

def load_img(img_path):
    return cv2.imread(img_path)

# display image from yolo ds
def display_dataset_img_bbox(image_path):
    # convert image path to label path
    label_path = image_path.replace('/images/', '/labels/')
    for ext in ['.jpeg', '.png', '.jpg']:
        if label_path.endswith(ext):
            label_path = label_path.replace(ext, '.txt')

    # Open the image and create ImageDraw object for drawing
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    with open(label_path, 'r') as f:
        for line in f.readlines():
            # Split the line into five values
            label, x, y, w, h = line.split(' ')

            # Convert string into float
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)

            # Convert center position, width, height into
            # top-left and bottom-right coordinates
            W, H = image.size
            x1 = (x - w / 2) * W
            y1 = (y - h / 2) * H
            x2 = (x + w / 2) * W
            y2 = (y + h / 2) * H

            # Draw the bounding box with red lines
            draw.rectangle((x1, y1, x2, y2),
                           outline=(255, 0, 0),  # Red in RGB
                           width=5)  # Line width
    image.show()


# Migrate Labels to Yolo format and create folder hierachy for yolo
def transform_data_to_yolo_format(ann_folder, data_folder):
    labels_folder = data_folder + "labels/"
    images_folder = data_folder + "images/"

    # Create the folder if it doesn't exist
    create_dir(data_folder)
    create_dir(labels_folder)
    create_dir(images_folder)

    for filename in os.listdir(ann_folder):
        name, extension = os.path.splitext(filename)
        if filename.endswith(".json"):
            # Load JSON data from the file
            json_file_path = os.path.join(ann_folder, filename)
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # Access objects and retrieving image size
            objects = data["objects"]

            img_width = data["size"]["width"]
            img_height = data["size"]["height"]

            # Write the space-separated string to a text file (YOLO FORMAT)
            label_path = labels_folder + name + '.txt'
            labels = []
            for obj in objects:
                points = obj["points"]["exterior"]
                if obj["classTitle"] == "Title":
                    labels.append(
                        convert_bbox_to_yolo_format(0, img_width, img_height, points[0][0], points[0][1], points[1][0],
                                                    points[1][1]))
                elif obj["classTitle"] == "Body text":
                    labels.append(
                        convert_bbox_to_yolo_format(1, img_width, img_height, points[0][0], points[0][1], points[1][0],
                                                    points[1][1]))

            with open(label_path, 'w', encoding='utf-8') as label_file:
                # Write objects in labels folder
                for idx, label in enumerate(labels):
                    label_str = ' '.join(map(str, label))
                    if idx == len(labels) - 1:
                        label_file.write(label_str)
                    else:
                        label_file.write(label_str + '\n')


# Fixing rotated image from exif info
def adjust_rotated_image(img_path):
    # Load the image and rotate it to correct orientation if needed
    image = Image.open(img_path)
    if hasattr(image, "_getexif"):  # Check if image has EXIF data (orientation)
        exif = image._getexif()
        if exif:
            orientation = exif.get(274, 1)  # Default orientation is 1 if not found in EXIF data
            rotation_angle = 0

            if orientation == 3:  # Rotate by 180 degrees
                rotation_angle = 180
            elif orientation == 6:  # Rotate clockwise by 90 degrees
                rotation_angle = -90
            elif orientation == 8:  # Rotate counterclockwise by 90 degrees
                rotation_angle = 90

            # Rotate the image if the orientation requires it
            if rotation_angle != 0:
                image = image.rotate(rotation_angle, expand=True)
    return image


# Displays a batch from the original Data
def display_batch_book_img_bbox(img_folder, ann_folder, batch_size):
    # Counter for the number of images displayed
    num_images_displayed = 0

    # Iterate over JSON files in the 'ann' folder
    for filename in os.listdir(ann_folder):
        if filename.endswith(".json"):
            # Load JSON data from the file
            json_file_path = os.path.join(ann_folder, filename)
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            # Access objects
            objects = data["objects"]

            # Find corresponding image file
            img_filename_base = os.path.splitext(filename)[0]  # Get filename without extension
            img_extensions = (".jpg", ".jpeg", ".png")
            img_path = None
            for ext in img_extensions:
                img_candidate_path = os.path.join(img_folder, img_filename_base + ext)
                if os.path.exists(img_candidate_path):
                    img_path = img_candidate_path
                    break

            if img_path is None:
                print(f"No corresponding image found for {filename}")
                continue

            # Adjusting rotated image to
            image = adjust_rotated_image(img_path)

            # Create figure and axis
            fig, ax = plt.subplots(1)

            # Display the image
            ax.imshow(image)

            # Define a function to draw polygons based on exterior points
            def draw_polygon(ax, points, color='r'):
                # Create and add the polygon patch to the plot
                polygon = patches.Polygon(points, closed=True, fill=None, edgecolor=color)
                ax.add_patch(polygon)

            # Define a function to draw rectangles based on exterior points
            def draw_rectangle(ax, points, color='b'):
                # Calculate the coordinates of the top-left corner and the width and height of the rectangle
                x1 = min(points[0][0], points[1][0])
                y1 = min(points[0][1], points[1][1])
                width = abs(points[0][0] - points[1][0])
                height = abs(points[0][1] - points[1][1])

                # Create and add the rectangle patch to the plot
                rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, fill=None)
                ax.add_patch(rect)

            # Iterate through objects and draw polygons or rectangles based on the geometry type
            for obj in objects:
                exterior_points = obj["points"]["exterior"]
                if obj["geometryType"] == "polygon":
                    draw_polygon(ax, exterior_points)
                elif obj["geometryType"] == "rectangle":
                    draw_rectangle(ax, exterior_points)

            # Show plot
            plt.show()

            # Increment the counter
            num_images_displayed += 1

            # Check if 5 images have been displayed
            if num_images_displayed >= batch_size:
                break


# converting x1,x2,y1,y2 to normalized x_c, y_c, w, h
def convert_bbox_to_yolo_format(title, img_width, img_height, xmin, ymin, xmax, ymax):
    """
    Convert bounding box coordinates to YOLO format.

    Args:
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        xmin (float): Top left x-coordinate of the bounding box.
        ymin (float): Top left y-coordinate of the bounding box.
        xmax (float): Bottom right x-coordinate of the bounding box.
        ymax (float): Bottom right y-coordinate of the bounding box.

    Returns:
        tuple: (x_center, y_center, width, height) in YOLO format.
    """
    # Calculate the center of the bounding box
    x_center = (xmin + xmax) / 2 / img_width
    y_center = (ymin + ymax) / 2 / img_height

    # Calculate the width and height of the bounding box
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height

    return title, x_center, y_center, width, height


def split_dataset(dataset_folder, train_size=672, val_size=83):
    i = 1
    for filename in os.listdir(dataset_folder + '/images/'):
        if os.path.isfile(os.path.join(dataset_folder + '/images/', filename)):
            name, extension = os.path.splitext(filename)

            if i < train_size:
                split = 'train'
            elif i < train_size + val_size:
                split = 'val'
            else:
                split = 'test'

            # Source paths
            source_image_path = dataset_folder + '/images/' + filename
            source_label_path = dataset_folder + '/labels/' + name + ".txt"

            # Destination paths
            target_image_folder = dataset_folder + '/images/' + split + '/' + filename
            target_label_folder = dataset_folder + '/labels/' + split + '/' + name + ".txt"

            # Copy files
            shutil.copy(source_image_path, target_image_folder)
            shutil.copy(source_label_path, target_label_folder)
            i += 1
