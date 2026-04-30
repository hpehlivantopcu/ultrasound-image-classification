from __future__ import annotations
import torch
import os
import json
from torchvision.io import read_image
from TumorDataset.utils.helper import load_annotations
from torchvision import tv_tensors
from math import ceil
from torchvision.transforms import v2 as T
import cv2


def find_files(root_dir, extensions):
    files = []
    for root, dirs, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(extensions):
                files.append(os.path.join(root, filename))
    return files


# def pad_to_minmum_size(image, min_size):
#     # print(image)
#     h, w = image.shape[1:]
#     h_diff = h - min_size
#     w_diff = w - min_size
#     h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
#     w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
#
#     if h_pad == 0 and w_pad == 0:
#         return image
#     else:
#         padded_image = T.functional.pad(image, [h_pad, w_pad])
#         return padded_image

# def pad_to_minmum_size(image, min_size):
#     # Get the height and width dimensions of the image tensor
#     h, w = image.size()[1:]
#     # print(f'original height: {h}')
#     # print(f'original width: {w}')
#
#     # Calculate the padding amounts for height and width
#     h_diff = h - min_size
#     w_diff = w - min_size
#     h_pad = ceil(abs(h_diff) / 2) if h_diff < 0 else 0
#     w_pad = ceil(abs(w_diff) / 2) if w_diff < 0 else 0
#     # print(f'h_pad: {h_pad}')
#     # print(f'w_pad: {w_pad}')
#
#     # Pad the image tensor if necessary
#     if h_pad == 0 and w_pad == 0:
#         return image
#     else:
#         # Pad the image tensor dynamically
#         padded_image = T.functional.pad(image, [w_pad, w_pad, h_pad, h_pad])
#         return padded_image


class TumorDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # Load JSON annotations
        self.imgs = list(sorted(find_files(root, '.jpg')))
        self.annotation_files = list(sorted(find_files(root, '.json')))

    def load_annotations(self, json_file):
        with open(json_file, 'r') as f:
            annotations = json.load(f)
        return annotations

    def __getitem__(self, idx):
        # Load image
        img_path = self.imgs[idx]
        annotation_path = self.annotation_files[idx]
        annotation = load_annotations(annotation_path)

        # img_path = os.path.join(self.root, annotation["imagePath"])
        # img = read_image(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = tv_tensors.Image(img)
        # img = T.to_tensor(read_image(img_path))
        # img = pad_to_minmum_size(img, 800)
        # height, width = img.shape[1:]
        # Extract annotations
        shapes = annotation["shapes"]
        boxes = []
        labels = []
        # masks = []
        for shape in shapes:
            # label = shape["label"]
            x_min, y_min = shape["points"][0]
            x_max, y_max = shape["points"][1]
            # x = x_min
            # y = y_min
            # width = x_max - x_min
            # height = y_max - y_min

            boxes.append([x_min, y_min, x_max, y_max])  # Format: (xmin, ymin, xmax, ymax)
            labels.append(1)

            # mask = torch.zeros((height, width), dtype=torch.uint8)
            # mask[int(y * height):int((y + h) * height), int(x * width):int((x + w) * width)] = 1
            # masks.append(mask)

        # Convert to tensor
        if self.transforms is not None:
            augmentations = self.transforms(image=img, bboxes=boxes)
            img = augmentations["image"]
            boxes = augmentations["bboxes"]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels)
        # masks = torch.stack(masks)

        # Generate additional target fields
        image_id = idx
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.uint8)  # Assuming no crowd instances
        target = {
            "boxes": boxes,
            "labels": labels,
            # "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        # Apply transformations
        # if self.transforms is not None:
        #     img, boxes, labels = self.transforms(img, boxes, labels)
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotation_files)
