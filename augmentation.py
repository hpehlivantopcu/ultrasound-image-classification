import albumentations as A
import cv2
from matplotlib import pyplot as plt
# from TumorDataset.utils.helper import get_transform

def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def plot_examples(images, bboxes=None):
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = 5

    for i in range(1, len(images)):
        if bboxes is not None:
            img = visualize_bbox(images[i - 1], bboxes[i - 1], class_name="Elon")
        else:
            img = images[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()


# From https://albumentations.ai/docs/examples/example_bboxes/
def visualize_bbox(img, bbox, class_name, color=(255, 0, 0), thickness=5):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = map(int, bbox)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    return img


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
    ])

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            min_area=1000,
            min_visibility=0.3,
            label_fields=[]
        )
    )


image = cv2.imread("Ultrasound-labeled/benign/1381668/20180115091839.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bboxes = [[
    219.24291497975707,
    131.39676113360323,
    531.7935222672064,
    298.1983805668016
]]

# Pascal_voc (x_min, y_min, x_max, y_max), YOLO, COCO
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = get_transform(train=True)

images_list = [image]
saved_bboxes = [bboxes[0]]
for i in range(15):
    augmentations = transform(image=image, bboxes=bboxes)
    augmented_img = augmentations["image"]

    if len(augmentations["bboxes"]) == 0:
        continue

    images_list.append(augmented_img)
    saved_bboxes.append(augmentations["bboxes"][0])

plot_examples(images_list, saved_bboxes)
