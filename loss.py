import torch
import torch.nn as nn
import torch.nn.functional as F
from util import readcfg

d = readcfg('cfg/yolond')
softmax = int(d['softmax'])
class_num = int(d['classes'])


class YOLOLoss(nn.Module):
    def __init__(self, side, num, sqrt, coord_scale, noobj_scale, use_gpu=True,vis=None,device=None):
        super(YOLOLoss, self).__init__()
        self.side = side
        self.num = num
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.sqrt = sqrt
        # self.use_gpu = torch.cuda.is_available()
        self.use_gpu = use_gpu
        self.device = torch.device('cuda:0')
        if device is not None:
            self.device = device
        self.vis = vis

    def compute_iou(self, box1, box2):
        """
        compute n predicted
        :param box1: [N,4(xmin,ymin,xmax,ymax)]
        :param box2: [1,4(xmin,ymin,xmax,ymax)]
        :return: [N]

        """
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # get the corrdinates of the intersection rectangle
        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        # Intersection area
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1, min=0)

        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        iou = inter_area / (b1_area + b2_area - inter_area)

        return iou

    def forward(self, preds, labels):
        return self.loss_1(preds, labels)

    def loss_1(self,preds,labels):
        '''
        preds: (tensor) size(batchsize,S,S,Bx5+20) [x,y,w,h,c]
        labels: (tensor) size(batchsize,S,S,Bx5+20)
        '''

        # print(preds.shape)
        # print(labels.shape)

        N = preds.size(0)
        bbox_size = self.num * 5
        cell_size = bbox_size + class_num

        obj_mask = labels[:, :, :, 4] > 0
        noobj_mask = labels[:, :, :, 4] == 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(labels)
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(labels)

        obj_pred = preds[obj_mask].view(-1, cell_size)
        box_pred = obj_pred[:, :bbox_size].contiguous().view(-1, 5)
        class_pred = obj_pred[:, bbox_size:]

        obj_label = labels[obj_mask].view(-1, cell_size)
        box_label = obj_label[:, :bbox_size].contiguous().view(-1, 5)
        class_label = obj_label[:, bbox_size:]

        # compute not containing loss
        noobj_pred = preds[noobj_mask].view(-1, cell_size)
        noobj_label = labels[noobj_mask].view(-1, cell_size)
        noobj_pred_mask = torch.zeros_like(noobj_pred, dtype=torch.bool)
        if self.use_gpu:
            noobj_pred_mask = noobj_pred_mask.to(self.device)
        for i in range(self.num):
            noobj_pred_mask[:, i * 5 + 4] = True
        noobj_pred_c = noobj_pred[noobj_pred_mask]
        noobj_label_c = noobj_label[noobj_pred_mask]
        noobj_loss = F.mse_loss(noobj_pred_c, noobj_label_c, reduction="sum")

        # object containing loss
        obj_response_mask = torch.zeros_like(box_label, dtype=torch.bool)
        if self.use_gpu:
            obj_response_mask = obj_response_mask.to(self.device)
        obj_not_response_mask = torch.zeros_like(box_label, dtype=torch.bool)
        if self.use_gpu:
            obj_not_response_mask = obj_not_response_mask.to(self.device)
        box_label_iou = torch.zeros(box_label.size())
        if self.use_gpu:
            box_label_iou = box_label_iou.to(self.device)

        s = 1/self.side
        for i in range(0, box_label.size(0), self.num):
            box1 = box_pred[i:i+self.num]
            box1_coord = torch.FloatTensor(box1.size())
            box1_coord[:, :2] = box1[:, :2] * s - 0.5 * box1[:, 2:4]
            box1_coord[:, 2:4] = box1[:, :2] * s + 0.5 * box1[:, 2:4]

            box2 = box_label[i].view(-1, 5)
            box2_coord = torch.FloatTensor(box2.size())
            box2_coord[:, :2] = box2[:, :2] * s - 0.5 * box2[:, 2:4]
            box2_coord[:, 2:4] = box2[:, :2] * s + 0.5 * box2[:, 2:4]
            iou = self.compute_iou(box1_coord[:, :4], box2_coord[:, :4])
            # print(iou.shape)
            # assert iou.shape[0] == self.num

            max_iou, max_index = iou.max(0)

            obj_response_mask[i + max_index] = True
            # obj_not_response_mask[i + 1 - max_index] = 1
            obj_not_response_mask[i:i + self.num] = True
            obj_not_response_mask[i + max_index] = False

            box_label_iou[i + max_index, torch.LongTensor([4])] = max_iou.data  # no grad

        # response loss
        box_pred_response = box_pred[obj_response_mask].view(-1, 5)
        box_label_response_iou = box_label_iou[obj_response_mask].view(-1, 5)
        box_label_response = box_label[obj_response_mask].view(-1, 5)
        response_loss = F.mse_loss(box_pred_response[:, 4], box_label_response_iou[:, 4], reduction="sum")
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_label_response[:, :2], reduction="sum")
        wh_loss = F.mse_loss(torch.sqrt(box_pred_response[:,2:4]), torch.sqrt(box_label_response[:,2:4]), reduction="sum")

        # not response loss
        box_pred_not_response = box_pred[obj_not_response_mask].view(-1, 5)
        box_label_not_response = box_label[obj_not_response_mask].view(-1, 5)
        box_label_not_response[:, 4] = 0
        not_response_loss = F.mse_loss(box_pred_not_response[:, 4], box_label_not_response[:, 4], reduction="sum")

        # class loss
        class_loss = F.mse_loss(class_pred, class_label, reduction="sum")

        total_loss = self.coord_scale*(xy_loss+wh_loss)+2.*response_loss+not_response_loss+1.0*self.noobj_scale*noobj_loss+class_loss

        return total_loss / N


if __name__ == '__main__':
    yololoss = YOLOLoss(14,2,1,5,.5)
    torch.manual_seed(1)
    pred = torch.rand(1, 14, 2, 30).cuda()
    target = torch.rand(pred.shape).cuda()
    loss = yololoss.loss_1(pred, target)
    loss_1 =yololoss.loss_2(pred,target)
    print(loss)  # 181.4906
    print(loss_1)
