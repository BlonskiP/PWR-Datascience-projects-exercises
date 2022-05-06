import torch
import torch.nn as nn
import torch.nn.functional as F

features = 832  # state length
fc1_out = 24
fc2_in = fc1_out
fc2_out = 32
out = 1


class DNeuralNetwork(nn.Module):
    def __init__(self):
        super(DNeuralNetwork, self).__init__()  # run base constructor

        self.full_connected_1 = nn.Linear(in_features=features, out_features=fc1_out)
        self.full_connected_2 = nn.Linear(in_features=fc2_in, out_features=fc2_out)
        self.out = nn.Linear(in_features=fc2_out, out_features=out)

    def forward(self, t: torch.Tensor):
    #    print('before reshape', t.shape)
        t = t.flatten(start_dim=1).float()
      #  print('reshape',t.shape)
        t = F.relu(self.full_connected_1(t))
        t = F.relu(self.full_connected_2(t))
        t = self.out(t)
        return t
