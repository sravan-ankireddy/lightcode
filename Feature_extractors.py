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
        elif self.NS_model == 4:
            out_size = d_model
            if self.mod == 'trx':
                input_size = input_size - self.reserve
                out_size = d_model - self.reserve
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = nn.ReLU()
            self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
            self.activation2 = nn.ReLU()
            self.FC3 = nn.Linear(d_model*3, out_size, bias=True)
        elif self.NS_model == 5:
            out_size = d_model * 3
            if self.mod == 'trx':
                input_size = input_size - self.reserve
                out_size = out_size + self.reserve
            self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
            self.activation1 = nn.ReLU()
            self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
            self.activation2 = nn.ReLU()
            self.FC3 = nn.Linear(out_size, d_model, bias=True)
        elif self.NS_model == 6:
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
            x1 = self.FC2(self.activation1(x1))
            x1 = self.FC3(self.activation2(x1))
            if self.mod == 'trx': # In the transmitter mode
                constantM = torch.ones_like(src[0])
                constantM[:,self.reserve:] = constantM[:,self.reserve:] * (-1)
                src1 = src * constantM
            elif self.mod == 'rec':
                src1 = src * (-1)
            elif self.mod == 'rec2': # belief update, not used here
                src1 = src * (-1)

            x2 = self.FC1(src1)########## => input_size to d_model dimension
            x2 = self.FC2(self.activation1(x2))
            x2 = self.FC3(self.activation2(x2))
            x = self.FC4(torch.cat([x1, x2], dim = 2))
        elif self.NS_model == 4:
            if self.mod == 'trx': # In the transmitter mode
                src1 = src[:,:,:self.reserve]
                x = self.FC1(src[:,:,self.reserve:])
                x = self.FC2(self.activation1(x))
                x = self.FC3(self.activation2(x))
                x = torch.cat([src1,x],dim=2)
            elif self.mod == 'rec': # In the decoder mode
                x = self.FC1(src)
                x = self.FC2(self.activation1(x))
                x = self.FC3(self.activation2(x))
            elif self.mod == 'rec2': # In the belief mode, not used here
                x = self.FC1(src)
                x = self.FC2(self.activation1(x))
                x = self.FC3(self.activation2(x))
        elif self.NS_model == 5:
            if self.mod == 'trx': # In the transmitter mode
                src1 = src[:,:,:self.reserve]
                x = self.FC1(src[:,:,self.reserve:])
                x = self.FC2(self.activation1(x))
                x = self.activation2(x)
                x = torch.cat([src1,x],dim=2)
                x = self.FC3(x)
            elif self.mod == 'rec': # In the decoder mode
                x = self.FC1(src)
                x = self.FC2(self.activation1(x))
                x = self.FC3(self.activation2(x))
            elif self.mod == 'rec2': # In the belief mode, not used here
                x = self.FC1(src)
                x = self.FC2(self.activation1(x))
                x = self.FC3(self.activation2(x))
        elif self.NS_model == 6:
            x1 = self.FC1(src)
            x1 = self.FC2(self.activation1(x1))
            if self.mod == 'trx': # In the transmitter mode
                constantM = torch.ones_like(src[0])
                constantM[:,self.reserve:] = constantM[:,self.reserve:] * - 1
                src1 = src * constantM
            elif self.mod == 'rec':
                src1 = src * -1
            elif self.mod == 'rec2': # belief update, not used here
                src1 = src * -1

            x2 = self.FC1(src1)########## => input_size to d_model dimension
            x2 = self.FC2(self.activation1(x2))
            x = self.FC4(torch.cat([x1, x2], dim = 2))
        return x


class ReducedFE(nn.Module):
    def __init__(self, mod, NS_model, input_size, d_model):
        super(ReducedFE, self).__init__()
        self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
        self.activation1 = nn.GELU()
        self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
        self.activation2 = nn.GELU()
        self.FC3 = nn.Linear(d_model*3, d_model*3, bias=True)
        self.activation3 = nn.GELU()
        self.FC4 = nn.Linear(d_model*3, d_model*3, bias=True)
        self.activation4 = nn.GELU()
        self.FC5 = nn.Linear(d_model*3, d_model, bias=True)
        
    def forward(self, src):
        x = self.FC1(src)
        x = self.FC2(self.activation1(x))
        x = self.FC3(self.activation2(x))
        x = self.FC4(self.activation3(x))
        x = self.FC5(self.activation4(x))
        return x
