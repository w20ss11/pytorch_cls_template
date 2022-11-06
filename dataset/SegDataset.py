from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pdb
from PIL import Image

class SegDataset(Dataset):

    def __init__(self, rgb_txt_path, msk_txt_path, transform=None):
        self.rgbs = open(rgb_txt_path, "r").readlines()
        self.msks = open(msk_txt_path, "r").readlines()
        assert(len(self.rgbs) == len(self.msks))
        # self.width = width
        # self.height = height
        self.transform = transform
        self.postprocess = transforms.Compose([
                                # transforms.ToPILImage(), # range(0,1) [c, h, w]  -> PIL: [h,w,c] (0,255)
                                # transforms.Resize((self.width, self.height)),
                                transforms.ToTensor(), # range(0,255) [h,w,c] -> tensor: range(0,1) [c,h,w]
                                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # 归一化到[-1, 1]，公式是：(x-0.5)/0.5
                            ])

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        rgb_path = self.rgbs[idx].strip()
        msk_path = self.msks[idx].strip()
        img = cv2.imread(rgb_path) # 像素顺序为BGR, (h,w,c)，读取出来即为array形式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 像素顺序为RGB, (h,w,c), array形式，PIL读取顺序和cv不一样，albumentation支持PIL格式，所以需要转
        msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE) # (h,w)
        
        if self.transform: #todo
            transformed = self.transform(image=img, mask=msk) #image must be numpy array type
            img = transformed["image"]
            msk = transformed["mask"]
        # img = Image.fromarray(img) # numpy->PIL.Image，用transforms.ToPILImage()比这句快很多
        img = self.postprocess(img) # c x args.width x args.height
        # msk =  transforms.ToTensor()(msk) # w x h -> 1 x w x h
        msk = msk / msk.max()
        sample = (img, msk)
        return sample


if __name__ == '__main__':
    import albumentations as A
    tf = A.Compose([ #todo random 
                A.HorizontalFlip(p=0.5), # 水平翻转
                A.RandomBrightnessContrast(p=0.5), # 随机选择图片的对比度和亮度
                A.Resize(256, 256)
            ])
    rgb_txt_path = "D:/code/pytorch_template/data/seg_txt/train_rgb.txt"
    msk_txt_path = "D:/code/pytorch_template/data/seg_txt/train_msk.txt"
    dataset = SegDataset(rgb_txt_path, msk_txt_path, transform=tf)
    img, msk = dataset.__getitem__(0)
    pdb.set_trace()
    print("img:", img.shape) # img: torch.Size([3, 256, 256])  范围：-1~1
    print("msk:", msk.shape) # label: 0 范围：0~1 使用：msk/msk.max()， ToTensor会除255
