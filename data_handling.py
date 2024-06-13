from helpers import create_dir
from config import config

data_folder = config.get('dataset_folder')
dataset_images = config.get_dataset_path('images')
dataset_labels = config.get_dataset_path('labels')

# Create a folder structure for YOLOv5 training
for folder in [dataset_images, dataset_labels]:
    for split in ['/train', '/val', '/test']:
        create_dir(folder + split)
