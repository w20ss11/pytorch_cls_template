from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pdb
from PIL import Image

class SegDataset(Dataset):

    def __init__(self, rgb_path, msk_path, width, height, transform=None):
        self.rgbs = open(rgb_path, "r").readlines()
        self.msks = open(msk_path, "r").readlines()
        assert(len(rgbs) == len(msks))
        self.width = width
        self.height = height
        self.transform = transform
        self.postprocess = transforms.Compose([
                                transforms.ToPILImage(), # range(0,1) [c, h, w]  -> PIL: [h,w,c] (0,255)
                                transforms.Resize((self.width, self.height)),
                                transforms.ToTensor(), # range(0,255) [h,w,c] -> tensor: range(0,1) [c,h,w]
                                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # 归一化到[-1, 1]，公式是：(x-0.5)/0.5
                            ])

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        rgb_path = self.rgbs[idx]
        msk_path = self.msks[idx]
        img = cv2.imread(jpg_path) # 像素顺序为BGR, (h,w,c)，读取出来即为array形式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# 像素顺序为RGB, (h,w,c), array形式，PIL读取顺序和cv不一样，albumentation支持PIL格式，所以需要转
        msk = cv2.imread(msk_path)
        
        if self.transform: #todo
            transformed = self.transform(image=img) #image must be numpy array type
            img = transformed["image"]
        # img = Image.fromarray(img) # numpy->PIL.Image，用transforms.ToPILImage()比这句快很多
        # sample = {'jpg_path':jpg_path, 'image': img, 'label': label}
        img = self.postprocess(img)
        sample = (img, label)
        return sample


if __name__ == '__main__':
    txt_path = "D:/code/pytorch_template/data/train.txt"
    dataset = CustomDataset(txt_path, 256, 256, None)
    img, label = dataset.__getitem__(0)
    print("img:", img.shape) # img: torch.Size([3, 256, 256])  范围：-1~1
    print("label:", label) # label: 0 范围：0~4
