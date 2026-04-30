from __future__ import annotations
import json
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
import os
from model_eval import get_transform


def traverse_folder_and_predict(folder_path, model):
    # Iterate over all the directories and files in the given folder
    no_tumors = 0
    no_tumor_images = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Construct the full path to the file
            file_path = os.path.join(root, file_name)
            json_file_path = file_path.replace(".jpg", "_pred.json")
            # Process the file (e.g., print its path)
            # if not file_path.endswith(".json") and not os.path.exists(json_file_path):
            if not file_path.endswith(".json") and not file_path.endswith(".DS_Store"):
                print(file_path)

                img = cv2.imread(file_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                transform = get_transform(train=False)
                transformed = transform(image=img)
                image_tensor = transformed['image']
                image_tensor = image_tensor.unsqueeze(0)

                with torch.no_grad():
                    predictions = model(image_tensor)

                boxes = predictions[0]['boxes'].cpu().detach().numpy()
                scores = predictions[0]['scores'].cpu().detach()
                if len(boxes) > 0:
                    boxes = boxes[0].tolist()
                    scores = scores[0].tolist()

                    res = {
                        "version": "5.0.1",
                        "flags": {},
                        "shapes": [
                            {
                                "label": "tumor",
                                "points": [
                                    [
                                        boxes[0],
                                        boxes[1],
                                    ],
                                    [
                                        boxes[2],
                                        boxes[3],
                                    ]
                                ],
                                "group_id": None,
                                "shape_type": "rectangle",
                                "flags": {}
                            }
                        ],
                        "imagePath": file_name,
                        "confidence_score": scores
                    }


                    with open(json_file_path, "w") as json_file:
                        json.dump(res, json_file)
                else:
                    no_tumors += 1
                    no_tumor_images.append(file_path)
    print(f"images where no tumors detected: {no_tumors}")
    print(no_tumor_images)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define the model architecture
    num_classes = 2
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Load the trained model weights
    model_path = 'output/resize_default_backbone_v3/best_model_segment_9__0.5663642546314341.pth'
    model.load_state_dict(torch.load(model_path))

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the appropriate device
    model.to(device)

    folder_to_traverse = 'External_test_set/'
    traverse_folder_and_predict(folder_to_traverse, model)
