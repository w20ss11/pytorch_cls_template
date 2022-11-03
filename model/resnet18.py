import torch
import torch.nn as nn
import torchvision.models as models

class ModelNet(nn.Module):
    def __init__(self, num_class):
        super(ModelNet, self).__init__()
        # todo feature
        self.model = models.resnet18(pretrained=False)
        self.model.load_state_dict(torch.load('data/resnet18-5c106cde.pth'))
        # self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)

    def forward(self, x):
        output = self.model(x)
        # output = torch.softmax(output, dim=0)
        return output

if __name__ == '__main___':
    x = torch.randn(1, 3, 32, 32)
    model = ModelNet()
    y = model(x)
    print(y.shape)
