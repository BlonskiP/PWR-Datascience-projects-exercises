import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, img_height, img_width, channels, actions_space_size,name="DQN"):
        super().__init__()
        self.name=name
        self.actions_space_size = actions_space_size
        self.input_size = (img_height, img_width, channels)
        self.fc1 = nn.Linear(
            in_features=self.input_size[0] * self.input_size[1] * self.input_size[2],
            out_features=24
        )
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=actions_space_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        qvalues = self.out(x)
        return qvalues
