import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, pdb

class FE(nn.Module):
    def __init__(self, mod, NS_model, input_size, d_model):
        super(FE, self).__init__()
        self.mod = mod
        self.NS_model = NS_model
        self.reserve = 3 + 8
        if self.NS_model == 0:
            self.FC1 = nn.Linear(input_size, d_model, bias=True)
        elif self.NS_model == 1:
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = F.relu
            self.FC2 = nn.Linear(d_model*3, d_model, bias=True)
        elif self.NS_model == 2:
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = nn.GELU()
            self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
            self.activation2 = nn.GELU()
            self.FC3 = nn.Linear(d_model*3, d_model, bias=True)
        elif self.NS_model == 3:
            self.FC1 = nn.Linear(input_size, d_model*2, bias=True)
            self.activation1 = nn.ReLU()
            self.FC2 = nn.Linear(d_model*2, d_model*2, bias=True)
            self.activation2 = nn.ReLU()
            self.FC3 = nn.Linear(d_model*2, d_model*2, bias=True)
            self.FC4 = nn.Linear(d_model*4, d_model, bias=True)

    def forward(self, src):
        if self.NS_model == 0:
            x = self.FC1(src)
        elif self.NS_model == 1:
            x = self.FC1(src)
            x = self.FC2(self.activation1(x))
        elif self.NS_model == 2:
            x = self.FC1(src)
            x = self.FC2(self.activation1(x))
            x = self.FC3(self.activation2(x))
        elif self.NS_model == 3:
            x1 = self.FC1(src)
            src1 = x1 * (-1)
            x1 = self.FC2(self.activation1(x1))
            x1 = self.FC3(self.activation2(x1))
            x = self.FC4(torch.cat([x1, src1], dim = 2))
        return x