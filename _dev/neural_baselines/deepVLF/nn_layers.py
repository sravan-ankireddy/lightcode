import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, pdb

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Transformer(nn.Module):
    def __init__(self, mod, input_size, block_size, d_model, N, heads, dropout, custom_attn=True, multclass = False):
        super(Transformer, self).__init__()
        self.mod = mod
        self.multclass = multclass
        self.block_size = block_size
        self.encoder = Encoder(mod, d_model, N, heads, dropout, input_size, custom_attn)
        if mod == "trx":
            d_model_reduced = int(d_model/4)
            self.out1 = nn.Linear(d_model, d_model_reduced)
            self.out2 = nn.Linear(d_model_reduced, 1)
        else:
            if multclass:
            	self.out = nn.Linear(d_model, 2**block_size)
            else:
            	self.out = nn.Linear(d_model, 2*block_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, src, mask, pe,idx,boundary):
        enc_out = self.encoder(src, mask, pe,idx,boundary)
        if self.mod == "rec":
            enc_out = self.out(enc_out)
        else:
            enc_out = self.out1(enc_out)
            enc_out = self.out2(enc_out)
        if self.mod == "rec":
            # decoders
            if self.multclass == False:
               batch = enc_out.size(0)
               numb_block = enc_out.size(1)
               enc_out = enc_out.contiguous().view(batch, numb_block*self.block_size,2)
               output = F.softmax(enc_out, dim=-1)
            else:
            	output = F.softmax(enc_out, dim=-1)
        else:
            # encoders
            output = enc_out
        return output

class Encoder(nn.Module):
    def __init__(self, mod, d_model, N, heads, dropout, input_size, customAttn= True):
        super(Encoder, self).__init__()
        if mod == 'trx': # In the transmitter mode
            flag_TX = 1
        else:
            flag_TX = 0
        self.Num_layers = N
        self.FC1 = nn.Linear(input_size, d_model*3, bias=True)
        self.FC2 = nn.Linear(d_model*3, d_model*3, bias=True)
        self.FC3 = nn.Linear(d_model*3, d_model*3, bias=True)
        self.FC4 = nn.Linear(d_model*3, d_model, bias=True)
        self.activation1 = F.relu
        self.activation2 = F.relu
        self.dropout = nn.Dropout(dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, flag_TX, dropout, custom_attn = customAttn), N)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, src, mask, pe,idx,tau_plus):
        ########## Variable Depth Feature extractor ############
        x = self.FC1(src.float())########## => input_size to d_model dimension
        x = self.FC2(self.activation1(x))
        ########## Change the network in a specific round ############
        if idx>tau_plus:
            x = self.FC3(self.activation1(x))
        x = self.FC4(self.activation2(x))
        ###########################################################################
        # Position encoding the src (dropout the output)
        x = pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) #layer normalization at the end

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, flag_TX,  dropout=0.1, norm_first = True, custom_attn = True):
        super(EncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5) # Original method used in Transformer
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5) # Original method used in Transformer
        if custom_attn:
        	self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffNet = FeedForward(d_model, d_ff=4 * d_model, dropout=dropout, act = 'relu')# linear -> act -> drop ->linear
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.custom = custom_attn

    def forward(self, x, mask):
        if self.norm_first:
            x2 = self.norm1(x)
            if self.custom:
            	x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=mask))
            else:
            	x = x + self.dropout1(self.self_attn(x2, x2, x2, attn_mask=mask)[0])
            x2 = self.norm2(x)
            x = x + self.dropout2(self.ffNet(x2))
        else:
            if self.custom:
            	x2 = self.self_attn(x, x, x, attn_mask=mask)
            else:
            	x2 = self.self_attn(x, x, x, attn_mask=mask)[0]
            x = x + self.dropout1(x2)
            x = self.norm1(x)
            x2 = self.ffNet(x)
            x = x + self.dropout2(x2)
            x = self.norm2(x)
        return x

#################################### Custom attention mechanism ###########################
def attention(q, k, v, d_k, attn_mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if attn_mask is not None:
        mask = attn_mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # pdb.set_trace()
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.FC = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, attn_mask=None, decoding=0):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * heads * sequenceLen * d_model/heads
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next

        scores = attention(q, k, v, self.d_k, attn_mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.FC(concat)

        return output

##################################### Custom FeedForward #################################
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1, act= 'relu'):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        if act == 'mish':
        	self.activation = F.mish
        elif act == 'silu':
        	self.activation = F.silu
        else:
        	self.activation = nn.GELU()

    def forward(self, x):
        x = self.dropout(self.activation(self.linear_1(x)))
        x = self.linear_2(x)
        return x








