import os
import argparse
import numpy as np
from datetime import datetime
import json
import pdb
import albumentations as A
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler

from model.unet import UNet
from dataset.segDataset import SegDataset
from utils.log import get_logger
from utils.averageMeter import AverageMeter


############################ PARAMS ################################################################
parser = argparse.ArgumentParser(description='params')
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--train_rgb_path', type=str, default="D:/code/pytorch_template/data/cls_txt/train.txt", help='')
parser.add_argument('--train_msk_path', type=str, default="D:/code/pytorch_template/data/cls_txt/train.txt", help='')
parser.add_argument('--test_rgb_path',  type=str, default="D:/code/pytorch_template/data/cls_txt/test.txt", help='')
parser.add_argument('--test_msk_path',  type=str, default="D:/code/pytorch_template/data/cls_txt/test.txt", help='')
parser.add_argument('--width',  type=int, default="256", help='the width of image')
parser.add_argument('--height', type=int, default="256", help='the height of image')
parser.add_argument('--batch_size', type=int, default="16", help='the batch size of per train data')
parser.add_argument('--init_lr', type=float, default="0.0001", help='learning rate')
parser.add_argument('--test_freq', type=int, default=1, help='the frequency to test model')
parser.add_argument('--save_path', type=str, default="./save", help='model save dir')
parser.add_argument('--save_freq', type=int, default="1", help='')
# parser.add_argument('--devides', type=int, default="5", help='')
parser.add_argument('--num_workers', type=int, default="0", help='')
parser.add_argument('--num_classes', type=int, default="5", help='')
args = parser.parse_args()

config_path = "./config.json"
fp_json = open(config_path, "r", encoding="utf-8")
json_content = json.load(fp_json)
gpu_ids = json_content['device']
date_str = datetime.now().strftime(r'%m%d_%H%M%S')
log_path = os.path.join(args.save_path, date_str, "log.txt")
logger = get_logger(log_path)

############################ SEED ################################################################
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

############################ SEED ################################################################
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
use_cuda = 0
if torch.cuda.is_available():
    use_cuda = 1
    cudnn.benchmark = True
# kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {'num_workers': args.num_workers}
device_ids = range(torch.cuda.device_count())

############################ MODEL ################################################################
transforms = A.Compose([ #todo random 
                A.HorizontalFlip(p=0.5), # 水平翻转
                A.RandomBrightnessContrast(p=0.5), # 随机选择图片的对比度和亮度
                A.Resize(args.height, args.height)
            ])
train_dataset = SegDataset(txt_path=args.train_data_path, transform=transforms)
test_dataset  = SegDataset(txt_path=args.test_data_path,  transform=transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
criterion = nn.BCEWithLogitsLoss()
model = UNet(n_channels=1, n_classes=1)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.init_lr, weight_decay=1e-8, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

############################ RESUME ###############################################################
start_epoch = 1
save_files = os.listdir(args.save_path)
save_models = {}
max_epoch = -1
for file in save_files:
    if file.endswith(".pth"):
        epoch_num = int(file[file.rfind('_')+1:file.rfind('.')])
        save_models[epoch_num] = os.path.join(args.save_path, file)
        max_epoch = epoch_num if epoch_num > max_epoch else max_epoch
if max_epoch != -1:
    checkpoint = torch.load(save_models[max_epoch])
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    start_epoch = epoch + 1

############################ TRAIN ################################################################
losses = AverageMeter("loss")
accuracy = AverageMeter("accu")
correct = 0
total = 0
if use_cuda and len(device_ids)>1:
    model=nn.DataParallel(model, device_ids=device_ids)
    print("now gpus are:" + device_ids)
elif use_cuda:
    model.cuda()
    print("now gpus are:" + str(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print("using cpu")
logger.info('start training!')

for epoch in range(start_epoch, args.epoch+1): #epoch 从1开始 loss不一致
    model.train()
    for i, (imgs, msks) in enumerate(train_loader):
        if use_cuda:
            imgs, msks = imgs.cuda(), msks.cuda()
        #清零
        outputs = model(images) # shape: batch_size x num_classes
        # print("labels:", labels)
        # pdb.set_trace()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        _, predicted = torch.max(outputs.cpu().data, 1)
        losses.update(loss.cpu().data.item())

        # total += labels.size(0)
        # correct += (predicted == labels).sum()
        # accuracy.update(100 * correct / total)
        # todo accu loss都是一個batch_size的
        if (i+1) % 2 == 0:
            logger.info('Epoch:%d/%d, Iter:%d/%d, Loss:%.4f, Accuracy:%.4f, lr:%.10f'% \
                (epoch, args.epoch, i+1, len(train_dataset)//args.batch_size, losses.avg, accuracy.avg, optimizer.state_dict()['param_groups'][0]['lr']))
        if epoch % args.save_freq == 0:
            model_path = os.path.join(args.save_path, "model_epoch_{}.pth".format(epoch))
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path)
    scheduler.step()

    model.eval()
    if epoch % args.test_freq == 0:
        for i, (images, labels) in enumerate(test_loader):
            if use_cuda:
                data,labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.cpu().data, 1)
            loss = criterion(outputs, labels)
            correct = (predicted == labels).sum()
            acc = 100 * correct / labels.size(0)
            if (i+1) % 2 == 0:
                logger.info('Epoch:%d/%d, Iter:%d/%d, Loss:%.4f, Accuracy:%.4f'% \
                    (epoch, args.epoch, i+1, len(test_dataset)//args.batch_size), loss, acc)

# if __name__ == '__main__':
#     main()
