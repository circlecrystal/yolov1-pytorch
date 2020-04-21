from torch.utils.data import DataLoader
import torch
from load_data import VocDataset
import time
from loss import YOLOLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from ftd_model import get_model_ft,load_model_trd
from mresnet import resnet50
from util import readcfg
from torchvision import transforms
import numpy as np
import torch.nn as nn
# from mmodels import mvgg
import os
#from adabound import adabound
import argparse
import sys
# side = 7
# num = 2
# classes = 20
# sqrt = 1
# noobj_scale = .5
# coord_scale = 5.
# object_scale = 1.
# class_scale = 1.
# batch_size = 16
# inp_size = 448
initial_lr = 0.001
momentum = 0.9
weight_decay = 5e-4
steps = [30, 40]
lr_scale = [0.1, 0.1]
num_epochs = 50

d = readcfg('cfg/yolond')
side = int(d['side'])
num = int(d['num'])
classes = int(d['classes'])
sqrt = int(d['sqrt'])
noobj_scale = float(d['noobj_scale'])
coord_scale = float(d['coord_scale'])
object_scale = float(d['object_scale'])
class_scale = float(d['class_scale'])
# batch_size = int(d['batch_size'])
batch_size = 16  # if gpu memory is enough, 16 ~ 64 is ok
inp_size = int(d['inp_size'])
# initial_lr = float(d['initial_lr'])
# momentum = float(d['momentum'])
# weight_decay = float(d['weight_decay'])
visualize = True
validate = True
vischange = False
save_final = False

# data_transforms = transforms.Compose([
#     # transforms.ToTensor(),
# ])

train_dataset = VocDataset('data/train.txt', side=side, num=num, input_size=inp_size, augmentation=False, transform=None)
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# train_dataset_size = len(train_dataset)
train_loader_size = len(trainloader)

test_dataset = VocDataset('data/voc_2007_test.txt', side=side, num=num, input_size=inp_size, augmentation=False, transform=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader_size = len(test_loader)


def train(target, sample_mode, enable_aug, window_size, batch_size,
          num_worker, num_boxes, num_cells_x, num_cells_y, conf_threshold,
          iou_threshold, lr, reg, lambda_coord, lambda_noobj,
          lambda_response, lambda_response_not, min_area, min_visibility,
          num_epochs, start_from, eval_level, output_dir):
    """
    Args:
        target (str): the target dataset
        sample_mode (boolean): report specs on the sampled trainset or the
            entire trainset
        enable_aug (boolean): enable data augmentation or not
        window_size (int): the width and height of the resized image
        batch_size (int): the batch size of SGD
        num_worker (int): the number of parallel processes
        num_boxes (int): the number of bounding boxes to detect per cell
        num_cells_x (int): the number of grid cells along the x-axis
        num_cells_y (int): the number of grid cells along the y-axis
        conf_threshold (float): When the confidence score of the box
            predictor is smaller than this conf_threshold, we remove the
            box predictor from the result.
        iou_threshold (float): When the IoU between the higher and lower
            confidence bounding boxes is larger than or equal to this IoU
            threshold value, the lower confidence bounding box is removed.
        lr (float): the learning rate
        reg (float): the regularization strength
        lambda_coord (float): the parameter for adjusting the strength of the
            object-related part of the loss function
        lambda_noobj (float): the parameter for adjusting the strength of the
            background-related part of the loss function
        lambda_response (float): the parameter for adjusting the strength of
            the object detection confidence part of the loss function
        lambda_response_not (float): the parameter for adjusting the penalty
            of the false positive bounding boxes of the loss function
        min_area (float): The minimum area of a bounding box. All bounding
            boxes hose visible area in pixels is less than this value will be
            removed.
        min_visibility (float): he minimum fraction of area for a bounding
            box to remain this box in list
        num_epochs (int): the number of epochs
        start_from (int): the number of epochs where the evaluation starts
        eval_level (str): the evaluation level "minimal", "compact" or "full"
        output_dir (str): the place for storing the trained model and the TB
            data of training process
    """
    # The device that tensors are stored (GPU if available)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        # Multiprocessing CUDA tensors is supported with "spawn" or "forkserver"
        # set_start_method("spawn")
    else:
        device = torch.device("cpu")

    # model = resnet50(
    #     num=num_boxes,
    #     side=num_cells_x,
    #     num_classes=20,
    #     softmax=False,
    #     detnet_block=False,
    #     downsample=False
    # ).to(device)
    model = get_model_ft("resnet50", pretrained=False).to(device)

    criterion = YOLOLoss(side=side, num=num, sqrt=sqrt, coord_scale=coord_scale, noobj_scale=noobj_scale, vis=None, device=device)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1, last_epoch=-1)

    for epoch in range(1, num_epochs + 1):
        running_loss = 0
        loss_avg = -1

        # Switch to training mode
        model.train()

        # Iterations
        for iteration, (inputs, targets) in enumerate(trainloader):
            # Get a batch of training data and targets
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            # Compute the Yolov1 training loss
            loss = criterion(outputs, targets)
            # Get the Yolov1 loss of this batch
            running_loss += loss.item()

            # Zeroing the accumulated gradients
            optimizer.zero_grad()
            # Backward pass
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            # Update the learnable parameters
            optimizer.step()

            # Update the exponentially averaged training loss
            if loss_avg < 0:
                loss_avg = loss.item()
            loss_avg = loss_avg*0.98+loss.item()*0.02

            if iteration % 5 == 0 or iteration + 1 == len(trainloader):
                print("Epoch [{}/{}], Iter [{}/{}] Loss: {:.4f}, average_loss: {:.4f}"\
                    .format(epoch, num_epochs, iteration, len(trainloader), loss.item(), loss_avg))

        if scheduler is not None and not dyn:
            scheduler.step()


def main():
    # # Initialize the learning rate and regularizer strength
    # lr = 10 ** np.random.uniform(-3, -6)
    # reg = 10 ** np.random.uniform(-5, 5)
    # lambda_coord = 10 ** np.random.uniform(-2, 2)
    # lambda_noobj = 10 ** np.random.uniform(-2, 2)
    # print("lr = {}".format(lr))
    # print("reg = {}".format(reg))

    train(
        target="VOC",
        sample_mode=False,
        enable_aug=False,
        window_size=448,
        batch_size=16,
        num_worker=4,
        num_boxes=2,
        num_cells_x=14,
        num_cells_y=14,
        conf_threshold=0.2,
        iou_threshold=0.4,
        lr=1e-3,
        reg=5e-4,
        lambda_coord=5,
        lambda_noobj=0.25,
        lambda_response=2,
        lambda_response_not=1,
        min_area=49,
        min_visibility=0.5,
        num_epochs=50,
        start_from=10,
        eval_level="lvl3",
        output_dir="training_outputs"
    )


if __name__ == "__main__":
    main()
