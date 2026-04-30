from __future__ import annotations
import json
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform(train, format='pascal_voc'):
    transforms = []
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        transforms.append(
            A.OneOf([
                A.BBoxSafeRandomCrop(erosion_rate=0.0),
                A.RandomSizedBBoxSafeCrop(height=500, width=500, erosion_rate=0.2)
            ], p=0.7)
        )
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.5),
            ], p=0.5),
        ])

    transforms.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format=format,
            min_area=1000,
            min_visibility=0.3,
            label_fields=[]
        )
    )


def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def find_files(root_dir, extensions):
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                files.append(os.path.join(root, filename))
    return files


def calculate_area(bboxes):
    areas = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        areas.append(area)
    return areas
