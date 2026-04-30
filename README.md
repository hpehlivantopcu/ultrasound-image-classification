augmentation.py - test out the Albumentations transform for both images nad bbox

detect_for_tumor.py - train object detection model using labeled data. Will indicate model with highest mAP50 scores in output folder

model_eval.py - using the best model predict the bbox for an image and then draw the original bbox and the predicted bbox next to each other

predict_bbox.py - given a folder name traverse the folder and all subfolders to find images to predict bboxes for. This script will output the prediction in a .json file in the same file location as the image. It will follow the json structure for the provided JSONs in Ultrasound-labeled.

code in detection folder are found here: https://github.com/pytorch/vision/tree/main/references/detection
