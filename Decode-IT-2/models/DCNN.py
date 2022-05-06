import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNN(nn.Module):
    def __init__(self, img_height, img_width, channels, actions_space_size,name="DCNN"):
        super().__init__()
        self.actions_space_size = actions_space_size

        self.conv1 = nn.Conv2d(4, 32, kernel_size=(8,8), stride=(4,4))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        inp = 3136
        self.f1 = nn.Linear(in_features=inp,out_features=512)
        self.out = nn.LazyLinear(out_features=actions_space_size)
        self.name = name

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x,start_dim=1)
        x = F.relu(self.f1(x))
        out = self.out(x.view(x.size(0), -1))

        return out