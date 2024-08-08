import torch
import torch.nn as nn
import math, pdb
from torch.autograd import Variable
import argparse
import numpy as np


# fixed PE
class PositionalEncoder_fixed(nn.Module):
    def __init__(self, lenWord=32, max_seq_len=200, dropout=0.0):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.lenWord)
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)


# learnable PE
class PositionalEncoder(nn.Module):
    def __init__(self, SeqLen=51, lenWord=64):
        super().__init__()
        self.lenWord = lenWord
        self.pe = torch.nn.Parameter(torch.Tensor(51, lenWord), requires_grad=True)
        self.pe.data.uniform_(0.0, 1.0)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :]




class Power_reallocate(torch.nn.Module):
    def __init__(self, args):
        super(Power_reallocate, self).__init__()
        self.args = args
        self.weight1 = torch.nn.Parameter(torch.Tensor(args.ell, 1), requires_grad=True)
        self.weight1.data.uniform_(1.0, 1.0)
        if args.seq_reloc == 1:
           self.weight2 = torch.nn.Parameter(torch.Tensor(args.T, 1), requires_grad=True)
           self.weight2.data.uniform_(1.0, 1.0)
           
            
    def forward(self, inputs, seq_order):
       # phase-level power allocation
        self.wt1 = torch.sqrt(self.weight1 ** 2 * (self.args.ell / torch.sum(self.weight1 ** 2)))
        if self.args.seq_reloc == 1:
            self.wt2 = torch.sqrt(self.weight2 ** 2 * ((self.args.T)/ torch.sum(self.weight2 ** 2)))
        inputs1 = inputs * self.wt1 # block_wise scaling
        if self.args.seq_reloc == 1:
            inputs1 = inputs1 * self.wt2[seq_order] # sequence_wise scaling

        return inputs1
