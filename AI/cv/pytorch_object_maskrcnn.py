"""

# Links
https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784
https://engineering.wootric.com/when-bert-meets-pytorch

"""
import copy
import json
import os
import numpy as np
import pandas as pd
from PIL import Image
from pypchutils.generic import create_logger
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.nn.functional as F

from AI.cv.detection import transforms as T
from AI.cv.detection.engine import train_one_epoch, evaluate
import AI.cv.detection.utils as utils

logger = create_logger(__name__)
logger.info("Successfully imported all the libraries")


def get_transform(train: bool):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB, because each color corresponds to
        # a different instance with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)  # convert the PIL Image into a numpy array

        obj_ids = np.unique(mask)  # instances are encoded as different colors

        obj_ids = obj_ids[1:]  # first id is the background, so remove it

        # split the color-encoded mask into a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs, boxes = len(obj_ids), []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes: int):

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # replace the pre-trained head with a new one
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # and replace the mask predictor with a new one
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def main(root: str, num_classes: int = 2, epochs: int = 10):
    """
    Args:
        num_classes: 2 means person and background
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = PennFudanDataset(root, get_transform(train=True))
    dataset_test = PennFudanDataset(root, get_transform(train=False))

    logger.info("Splitting the dataset in train and test sets...")
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    logger.info("Defining the data loaders..")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4, collate_fn=utils.collate_fn
    )

    model = get_model_instance_segmentation(num_classes)
    model.to(device)
    logger.info("Successfully loaded the model")

    # construct an optimizer, and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()  # update the learning rate
        evaluate(model, data_loader_test, device=device)  # evaluate on the test dataset

    logger.info("Successfully completed training")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=f"{os.environ.get('HOME')}/Google Drive/xheng/data/PennFudanPed")
    args = vars(parser.parse_args())
    logger.info("Cmd line args:\n{}".format(json.dumps(args, sort_keys=True, indent=4)))

    main(**args)
    logger.info("ALL DONE!\n")
