import torch
import torch.nn as nn
import torchvision.models as models

class ModelNet(nn.Module):
    def __init__(self, num_class, criterion):
        super(ModelNet, self).__init__()
        # todo feature
        self.model = models.resnet18(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_class)
        self.criterion = criterion

    def forward(self, x, gt=None):
        output = self.model(x)
        output = torch.softmax(output, dim=0)
        loss = None
        if not gt is None:
            loss = self.criterion(output, gt)
        return output, loss

if __name__ == '__main___':
    x = torch.randn(1, 3, 32, 32)
    model = ModelNet()
    y = model(x)
    print(y.shape)
