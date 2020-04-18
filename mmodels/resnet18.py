import torch
import torchvision


class YoloConv(torch.nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.yolo_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(512, depth, 3, 1, 1),
            torch.nn.BatchNorm2d(depth),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Formating the output tensor
        x = self.yolo_conv(x)
        x = x.permute(0, 2, 3, 1)

        return x


class YoloFC(torch.nn.Module):
    def __init__(self, num_cells_x, num_cells_y, depth):
        """
        Args:
            num_cells_x (int): the number of cells along x-axis of Yolo
            num_cells_y (int): the number of cells along y-axis of Yolo
            depth (int): the number of elements in each tensor cell
        """
        super().__init__()

        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        self.depth = depth

        self.yolo_fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(1),
            torch.nn.Linear(512, num_cells_x * num_cells_y * self.depth),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        # Formating the output tensor
        x = self.yolo_fc(x)
        x = x.view(-1, self.num_cells_x, self.num_cells_y, self.depth)

        return x


class ResNet18(torch.nn.Module):
    """The ResNet18 backbone.
    """

    def __init__(self, num_cells_x=7, num_cells_y=7, num_boxes=2,
                 num_categories=20):
        """
        Args:
            num_cells_x (int): the number of cells along x-axis of Yolo
            num_cells_y (int): the number of cells along y-axis of Yolo
            num_boxes (int): the number of object boxes to localize
            num_categories (int): the number of object categories to detect
        """
        super().__init__()

        # The number of elements in each tensor cell
        depth = 5 * num_boxes + num_categories

        # Import the backbone net
        resnet18 = torchvision.models.resnet18(pretrained=False)

        # Extracting the layers
        self.layer1 = torch.nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )
        self.layer2 = resnet18.layer1
        self.layer3 = resnet18.layer2
        self.layer4 = resnet18.layer3
        self.layer5 = resnet18.layer4
        self.yololayer = YoloFC(num_cells_x, num_cells_y, depth)
        # self.yololayer = YoloConv(depth)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.yololayer(x)

        return x
