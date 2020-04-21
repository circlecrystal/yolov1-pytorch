from torch.utils.data import DataLoader
import torch
from load_data import VocDataset
import time
from loss import YOLOLoss
import torch.optim as optim
from torch.optim import lr_scheduler
from ftd_model import get_model_ft,load_model_trd
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


def train_model(model, criterion, optimizer, scheduler, num_epochs, dyn=False):
    since = time.time()
    # best_model_wts = copy.deepcopy(model.state_dict())
    best_test_loss = np.inf
    lr = initial_lr
    s = 0

    if dyn:
        print('using dynamic learning rate')
    for epoch in range(start_epoch+1, num_epochs):
        model.train()
        # if scheduler is None:
        #     # for iteration, step in enumerate(steps):
        #     #     if epoch == step:
        #     #         lr = lr * lr_scale[iteration]
        #     #         break
        #     if s < len(steps) and steps[s] == epoch:
        #         lr = lr * lr_scale[s]
        #         s += 1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # else:
        #     scheduler.step()
        #     lr = scheduler.get_lr()
        if scheduler is not None and not dyn:
            lr = scheduler.get_lr()

        print('Epoch {}/{}, lr:{}'.format(epoch + 1, num_epochs, lr))
        print('-' * 16)

        running_loss = 0.0

        loss_avg = -1

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

        # print('\nEpoch[{}], average loss: {:.4f}\n'.format(epoch+1, running_loss/train_loader_size))

    #     if s < len(steps) and (epoch+1) == steps[s]:
    #         print("save {}, step {}, learning rate {}".format(model_name, epoch+1, lr))
    #         torch.save({'epoch': epoch, 'lr': lr, 'model': model.state_dict()}, backupdir+"{}_step_{}.pth".format(model_name, epoch+1))
    #         s += 1

    #     # validation
    #     if validate:
    #         validation_loss = 0.0
    #         model.eval()
    #         for iteration, (imgs, target) in enumerate(test_loader):
    #             imgs = imgs.to(device)
    #             target = target.to(device)

    #             out = model(imgs)
    #             loss = criterion(out, target)
    #             validation_loss += loss.item()

    #         validation_loss /= test_loader_size
    #         if scheduler is not None and dyn:
    #             scheduler.step(validation_loss)

    #         if visualize:
    #             vis.plot_many_stack({'train': running_loss / train_loader_size, 'val': validation_loss})
    #         print('validation loss:{}'.format(validation_loss))

    #         if best_test_loss > validation_loss:
    #             best_test_loss = validation_loss
    #             print('epoch%d, get best test loss %.5f' % (epoch+1, best_test_loss))
    #             torch.save({'epoch': epoch, 'best_loss':best_test_loss, 'lr': lr, 'model': model.state_dict()}, backupdir+'{}_best.pth'.format(model_name))

    # # end
    # if num_epochs > 20 or save_final:
    #     torch.save({'epoch':num_epochs-1, 'lr':lr, 'model':model.state_dict()}, backupdir+'{}_final.pth'.format(model_name))
    # time_elapsed = int(time.time() - since)
    # h = time_elapsed // 3600
    # m = (time_elapsed % 3600) // 60
    # s = time_elapsed % 60
    # print('{} epochs, spend {}h:{}m:{:.0f}s'.format(num_epochs, h, m, s))


def arg_parse():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-w", dest="weight", help="load weight", type=str)
    arg_parser.add_argument("-n", dest="net", default="resnet50", help="backbone net", type=str)
    arg_parser.add_argument("-env",dest="env", help="visdom environment",type=str)

    return arg_parser.parse_args()


args = arg_parse()
model_name = args.net
env = args.env
if env is None:
    print('no visdom-environment specified, visualization off')
    visualize = False
    # sys.exit(1)
vis = None
if visualize:
    from visualize import Visualizer
    vis = Visualizer(env='{}{}_{}'.format(model_name, env, time.strftime('%m%d%H%M')))

start_epoch = -1
if args.weight is not None:
    model_ft, start_epoch, lr = load_model_trd(model_name, args.weight)
    print('weight loaded', 'epoch:', start_epoch+1, 'lr:', lr)
else:
    print('no weight specified, training from 0')
    model_ft = get_model_ft(model_name)
    # sys.exit(1)
assert model_ft is not None
# print(model_ft)

model_ft.to(device)

# vis = Visualizer('cood2loss_2_cmp_{}'.format(time.strftime('%m%d%H%M')))
criterion = YOLOLoss(side=side, num=num, sqrt=sqrt, coord_scale=coord_scale, noobj_scale=noobj_scale, vis=vis,device=device)

# params=[]
# params_dict = dict(model_ft.named_parameters())
# for key,value in params_dict.items():
#     print(key,value.shape)
#     if key.startswith('features'):
#         params += [{'params':[value],'lr':initial_lr*1}]
#     else:
#         params += [{'params':[value],'lr':initial_lr}]
optimizer_ft = optim.SGD(model_ft.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
# optimizer_ft = adabound.AdaBound(model_ft.parameters(),lr=1e-3,final_lr=0.1)
# scheduler_dyn = lr_scheduler.ReduceLROnPlateau(optimizer_ft,mode='min',patience=3,verbose=True)
if start_epoch != -1:
    for group in optimizer_ft.param_groups:
        group.setdefault('initial_lr', initial_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=steps, gamma=0.1, last_epoch=start_epoch)

if not os.path.exists('backup'):
    os.mkdir('backup')
backupdir = 'backup/db07/'
if not os.path.exists(backupdir):
        os.mkdir(backupdir)
train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=num_epochs,dyn=False)
