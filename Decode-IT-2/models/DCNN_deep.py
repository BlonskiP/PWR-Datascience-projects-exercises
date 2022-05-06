import torch
import torch.nn as nn
import torch.nn.functional as F

class DCNN_DEEP(nn.Module):
    def __init__(self, img_height, img_width, channels, actions_space_size,name="DCNN"):
        super().__init__()
        self.actions_space_size = actions_space_size

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False),
                                 nn.ReLU(True),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
                                 nn.ReLU(True),
                                 nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
                                 nn.ReLU(True),
                                 nn.Conv2d(64, 1024, kernel_size=7, stride=1, bias=False),
                                 nn.ReLU(True)
                                 )
        self.streamA = nn.Linear(512, actions_space_size)
        self.streamV = nn.Linear(512, 1)

        inp = 2592
        self.f1 = nn.Linear(in_features=inp,out_features=254)
        self.out = nn.LazyLinear(out_features=actions_space_size)
        self.name = name

    def forward(self, x):
        x = self.cnn(x)
        sA, sV = torch.split(x, 512, dim=1)
        sA = torch.flatten(sA, start_dim=1)
        sV = torch.flatten(sV, start_dim=1)
        sA = self.streamA(sA)  # (B,4)
        sV = self.streamV(sV)  # (B,1)
        # combine this 2 values together
        Q_value = sV + (sA - torch.mean(sA, dim=1, keepdim=True))
        return Q_value
