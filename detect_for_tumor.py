from __future__ import annotations
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from detection.engine import train_one_epoch, evaluate
from detection import utils


from TumorDataset import TumorDataset
from TumorDataset.utils.helper import get_transform

if __name__ == '__main__':
    root_dir = 'Ultrasound-labeled'

    dataset = TumorDataset(root_dir, transforms=get_transform(train=True))
    dataset_test = TumorDataset(root_dir, transforms=get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-70])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-30:])

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=utils.collate_fn
    )

    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    num_classes = 2

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    # Define device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Define optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.6, weight_decay=0.0)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    best_map50 = 0.0
    model_folder = 'output/resize_default_backbone_v2'
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        torch.save(model.state_dict(), f'{model_folder}/model_segment_{str(epoch)}.pth')
        map50 = evaluate(model, test_dataloader, device=device)
        mAP50 = map50.coco_eval['bbox'].stats[0]
        # mAP50 = map50.coco_eval['boxes'].stats[0]
        #
        # Save the model if mAP50 improves
        if mAP50 > best_map50:
            best_map50 = mAP50
            torch.save(model.state_dict(), f'{model_folder}/best_model_segment_{str(epoch)}__{mAP50}.pth')
        print(f'Epoch [{epoch}/{num_epochs}], mAP50: {mAP50}, Best mAP50: {best_map50}')

    # Save trained model
    # torch.save(model.state_dict(), 'final_model_segment.pth')
