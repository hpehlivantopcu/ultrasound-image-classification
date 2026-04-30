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

# from TumorDataset.utils.helper import get_transform

def get_transform(train):
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
        transforms
    )


def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def draw_boxes_side_by_side(image, boxes, original_boxes):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Draw box on the first image
    axs[0].imshow(image)
    xmin, ymin, xmax, ymax = boxes[0]
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=3, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)
    axs[0].set_title('Predicted Box')

    # xmin, ymin, xmax, ymax = boxes[1]
    # rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=3, edgecolor='r', facecolor='none')
    # axs[0].add_patch(rect)
    # axs[0].set_title('Predicted Box')

    # Draw box on the second image
    if len(original_boxes) > 0:
        axs[1].imshow(image)
        xmin, ymin, xmax, ymax = original_boxes[0]
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=3, edgecolor='y', facecolor='none')
        axs[1].add_patch(rect)
        axs[1].set_title('Original Box')

    plt.show()


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

    root_path = 'External_test_set/benign/211006974/1.2.826.0.1.3680043.2.461.12538571.1755266164'
    # root_path = 'Ultrasound/benign/13205272/20200803083401' Ultrasound/benign/13162838/20190821094655
    image_path = f'{root_path}.jpg'
    annotation_path = f'{root_path}.json'
    json_data = load_annotations(annotation_path)
    # print(json_data) Ultrasound/benign/13205272/20200803083401.jpg Ultrasound/malignant/1396156/20180427084800
    points_list = []
    shapes = json_data["shapes"]
    for shape in shapes:
        x_min, y_min = shape["points"][0]
        x_max, y_max = shape["points"][1]

        points_list.append([x_min, y_min, x_max, y_max])

    print(points_list)


    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define the device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    transform = get_transform(train=False)  # .to(device)
    # Apply the transformation to the image
    transformed = transform(image=img)
    image_tensor = transformed['image']
    image_tensor_copy = transformed['image'].clone()

    # Add a batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    # Print the predicted boxes
    print(predictions)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denorm = A.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        max_pixel_value=1.0
    )

    img_np = image_tensor_copy.permute(1, 2, 0).cpu().numpy()

    image_tensor = denorm(image=img_np)["image"]

    image = T.ToPILImage()(image_tensor)

    # Extract boxes and labels
    boxes = predictions[0]['boxes'].cpu().detach().numpy()
    labels = predictions[0]['labels'].cpu().detach()
    scores = predictions[0]['scores'].cpu().detach()
    boxes = boxes.tolist()
    scores = scores.tolist()
    print(boxes)
    print(scores)

    # original_size = (768, 576)
    draw_boxes_side_by_side(image, boxes, points_list)
    if len(boxes) > 0:
        draw_boxes_side_by_side(image, boxes, points_list)
    else:
        print("nothing detected")
