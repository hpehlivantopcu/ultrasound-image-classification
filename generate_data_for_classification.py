from __future__ import annotations
import os
import cv2
import json


def crop_save_image(image_file, x_min, y_min, x_max, y_max, my_index):
    label = 'benign' if 'benign' in image_file else 'malignant'
    tmp = image_file.split('/')[-2:]
    croped_file_name = '__'.join(tmp)

    img = cv2.imread(image_file)
    # print(f'external_data/{label}/{my_index}__{croped_file_name}')

    # Crop the image
    cropped_img = img[int(y_min):int(y_max), int(x_min):int(x_max)]

    # Save the cropped image
    save_file_name = f'external_data/{label}/{my_index}__{croped_file_name}'
    cv2.imwrite(save_file_name, cropped_img)

    print(f"Saved cropped image to: {save_file_name}")


def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annotations = json.load(f)
    return annotations


def traverse_folder_and_predict(folder_path):
    # Iterate over all the directories and files in the given folder
    no_tumors = 0
    no_tumor_images = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            # Construct the full path to the file
            file_path = os.path.join(root, file_name)
            # json_file_path = file_path.replace(".jpg", ".json")
            # Process the file (e.g., print its path)
            # if not file_path.endswith(".json") and not os.path.exists(json_file_path):
            if file_path.endswith(".json"):
                # print(file_path)
                annotation = load_annotations(file_path)
                shapes = annotation["shapes"]
                image_path = file_path.replace('.json', '.jpg')

                for i, shape in enumerate(shapes):
                    # label = shape["label"]
                    x_min, y_min = shape["points"][0]
                    x_max, y_max = shape["points"][1]
                    crop_save_image(image_path, x_min, y_min, x_max, y_max, i)




if __name__ == '__main__':
    # print('hi')
    root_dir = 'Ultrasound-labeled'
    traverse_folder_and_predict(root_dir)

