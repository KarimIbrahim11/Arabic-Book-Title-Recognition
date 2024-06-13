import os
import cv2
import json
import numpy as np


def cat_json(json_file):
    # Load JSON data
    with open(json_file, "r", encoding="utf-8") as json_file:
        json_data = json.load(json_file)
    # Print JSON structure
    print("JSON Structure:")
    print(json.dumps(json_data, indent=4))


def create_dir(pth):
    if not os.path.exists(pth):
        os.makedirs(pth)
        print(f"Folder '{pth}' created.")


def cat_pth(pth, title=None):
    if title:
        print(f"{title} : {pth}")
    else:
        print(f"{pth}")
