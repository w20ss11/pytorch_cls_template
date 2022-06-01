from torch.utils.data import Dataset
from torchvision import transforms
import cv2
# from PIL import Image

class CustomDataset(Dataset):

    def __init__(self, txt_path, width, height, transform=None):
        fp = open(txt_path, "r")
        self.inf = fp.readlines()
        self.width = width
        self.height = height
        self.transform = transform
        self.preprocess = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((self.width, self.height)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                    ])

    def __len__(self):
        return len(self.inf)

    def __getitem__(self, idx):
        infos = self.inf[idx].split()
        jpg_path = infos[0]
        label = int(infos[1])
        img = cv2.imread(jpg_path)
        img = self.preprocess(img)

        if self.transform:
            img = self.transform(img)

        # sample = {'jpg_path':jpg_path, 'image': img, 'label': label}
        sample = (img, label)

        return sample


if __name__ == '__main__':
    txt_path = "D:/code/pytorch_template/data/train.txt"
    dataset = CustomDataset(txt_path, 32, 32)
    data = dataset.__getitem__(0)
    print("img path is ", data["jpg_path"])
    print("img shape is ", data["image"].shape)
    print("img label is ", data["label"])