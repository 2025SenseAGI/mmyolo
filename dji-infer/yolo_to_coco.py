import os
import json
from PIL import Image

# Set your paths
# images_dir = 'images'
# labels_dir = 'labels'
# output_json = 'mmyolo_annotations.json'
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def yolo_to_coco_bbox(bbox, img_w, img_h):
    x_center, y_center, w, h = bbox
    x = (x_center - w / 2) * img_w
    y = (y_center - h / 2) * img_h
    w = w * img_w
    h = h * img_h
    return [x, y, w, h]

def yolo_to_coco(images_dir, labels_dir, output_json):

    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(class_names)]
    ann_id = 1

    for img_id, img_name in enumerate(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(images_dir, img_name)
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        img = Image.open(img_path)
        width, height = img.size

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                bbox = yolo_to_coco_bbox([x_center, y_center, w, h], width, height)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                })
                ann_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)

yolo_to_coco('./helmet/train/images', './helmet/train/labels', './helmet/annotations/train_annotation.json')
yolo_to_coco('./helmet/valid/images', './helmet/valid/labels', './helmet/annotations/valid_annotation.json')