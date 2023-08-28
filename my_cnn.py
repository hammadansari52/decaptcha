import torch
import torch.nn as nn

class MyCNNGAP(nn.Module):
  def __init__(self):
    super(MyCNNGAP, self).__init__()

    self.s1 = nn.Sequential(nn.BatchNorm2d(1),
                            nn.Conv2d(1, 16, 3),
                            nn.ReLU(),
                            nn.MaxPool2d(3),
                            nn.Conv2d(16, 32, 3, padding=1),
                            nn.ReLU(),
                            nn.BatchNorm2d(32),
                            nn.Conv2d(32,24,3, padding=1),
                            nn.AdaptiveAvgPool2d((1,1))
    )

  def forward(self, x):
    x = self.s1(x)
    x = torch.flatten(x, start_dim=1)
    return nn.functional.softmax(x,-1) 