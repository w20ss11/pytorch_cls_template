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
from torchvision import transforms
from PIL import Image

from model.unet import UNet
from dataset.SegDataset import SegDataset
from utils.log import get_logger
from utils.AverageMeter import AverageMeter
from utils.metric import confusion_matrix, evaluate
from utils.common import DeNormalize


############################ PARAMS ################################################################
parser = argparse.ArgumentParser(description='params')
parser.add_argument('--epoch', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--train_rgb_path', type=str, default="D:/code/pytorch_template/data/seg_txt/train_rgb.txt", help='')
parser.add_argument('--train_msk_path', type=str, default="D:/code/pytorch_template/data/seg_txt/train_msk.txt", help='')
parser.add_argument('--test_rgb_path',  type=str, default="D:/code/pytorch_template/data/seg_txt/train_rgb.txt", help='')
parser.add_argument('--test_msk_path',  type=str, default="D:/code/pytorch_template/data/seg_txt/train_msk.txt", help='')
parser.add_argument('--width',  type=int, default="256", help='the width of image')
parser.add_argument('--height', type=int, default="256", help='the height of image')
parser.add_argument('--batch_size', type=int, default="4", help='the batch size of per train data')
parser.add_argument('--init_lr', type=float, default="0.0001", help='learning rate')
parser.add_argument('--threshold', type=float, default="0.5", help='threshold')
parser.add_argument('--test_freq', type=int, default=1, help='the frequency to test model')
parser.add_argument('--save_path', type=str, default="./save", help='model save dir')
parser.add_argument('--save_freq', type=int, default="1", help='')
# parser.add_argument('--devides', type=int, default="5", help='')
parser.add_argument('--num_workers', type=int, default="0", help='')
parser.add_argument('--num_classes', type=int, default="2", help='')
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
tf = A.Compose([ #todo random 
                A.HorizontalFlip(p=0.5), # 水平翻转
                A.RandomBrightnessContrast(p=0.5), # 随机选择图片的对比度和亮度
                A.Resize(args.height, args.height)
            ])
train_dataset = SegDataset(rgb_txt_path=args.train_rgb_path, msk_txt_path=args.train_msk_path, transform=tf)
test_dataset  = SegDataset(rgb_txt_path=args.train_rgb_path, msk_txt_path=args.train_msk_path, transform=tf)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
criterion = nn.BCEWithLogitsLoss()
model = UNet(in_ch=3, out_ch=1)
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
miou = AverageMeter("miou")
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
    conf_mat = np.zeros((args.num_classes, args.num_classes)).astype(np.int64) # 每个epoch的混淆矩阵是累加的
    model.train()
    for i, (imgs, msks) in enumerate(train_loader):
        if use_cuda:
            imgs, msks = imgs.cuda(), msks.cuda()
        # 清零 bp
        outputs = model(imgs) # shape: batch_szie x c x w x h
        outputs = torch.sigmoid(outputs) # 分割这里只支持二分类，多分类需修改mask和model输出为bXclassesXwXh和softmax(outputs)
        # print("imgs:", imgs.shape) # batch_szie x c x w x h
        # print("msks:", msks.shape) # batch_szie x w x h
        # print("outputs:", outputs.shape) # batch_szie x 1 x w x h
        loss = criterion(outputs.squeeze(), msks.squeeze())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 计算混淆矩阵和acc，iou
        losses.update(loss.cpu().data.item())
        preds = (outputs.data.cpu().numpy().squeeze() > args.threshold).astype(np.uint8) # batch_size x w x h
        msks = msks.data.cpu().numpy().squeeze().astype(np.uint8) # batch_size x w x h
        conf_mat += confusion_matrix(pred=preds.flatten(), label=msks.flatten(), num_classes=args.num_classes)
        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_mean_IoU, train_kappa = evaluate(conf_mat)
        accuracy.update(train_acc)
        miou.update(train_mean_IoU)
        # 打印
        if (i+1) % 2 == 0:
            logger.info('Epoch:%d/%d, Iter:%d/%d, Loss:%.4f, Accuracy:%.4f, miou:%.4f, lr:%.10f'% \
                (epoch, args.epoch, i+1, len(train_dataset)//args.batch_size, losses.avg, accuracy.avg, miou.avg, optimizer.state_dict()['param_groups'][0]['lr']))
        # 存模型
        if epoch % args.save_freq == 0:
            model_path = os.path.join(args.save_path, "segm_odel_epoch_{}.pth".format(epoch))
            state = {'model': model.state_dict(), 'oiptmizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(state, model_path)
    scheduler.step()


    model.eval()
    resore_transform = transforms.Compose([DeNormalize([.5, .5, .5], [.5, .5, .5])])
    if epoch % args.test_freq == 0:
        conf_mat = np.zeros((args.num_classes, args.num_classes)).astype(np.int64)
        for i, (imgs, msks) in enumerate(test_loader):
            if use_cuda:
                imgs,msks = imgs.cuda(), msks.cuda()
            outputs = model(imgs)
            outputs = torch.sigmoid(outputs) # 分割这里只支持二分类，多分类需修改mask和model输出为bXclassesXwXh和softmax(outputs)
            loss = criterion(outputs.squeeze(), msks.squeeze())
            
            # 计算混淆矩阵和acc，iou
            preds = (outputs.data.cpu().numpy().squeeze() > args.threshold).astype(np.uint8) * 255
            msks = msks.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += confusion_matrix(pred=preds.flatten(), label=msks.flatten(), num_classes=args.num_classes)
            val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_mean_IoU, val_kappa = evaluate(conf_mat)
            # 可视化
            batch_idx = 0
            # img_pil = resore_transform(imgs[batch_idx][0].cpu().repeat(0, 3)) # 只取了每个batch的第1张
            preds_pil = Image.fromarray(preds[batch_idx].astype(np.uint8)).convert('L')
            # img_pil.save(os.path.join("", 'img_batch_%d_%d.jpg' % (i, batch_idx)))
            preds_pil.save(os.path.join("", 'label_%d_%d.png' % (i, batch_idx)))
            # 打印
            if (i+1) % 2 == 0:
                logger.info('Epoch:%d/%d, Iter:%d/%d, Loss:%.4f, Accuracy:%.4f, miou:%.4f'% \
                    (epoch, args.epoch, i+1, len(test_dataset)//args.batch_size, loss.cpu().item(), val_acc, val_mean_IoU))

# if __name__ == '__main__':
#     main()
