############# Network layers from top to bottom in a hiearchical order #############################
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, pdb
from torch.nn import TransformerEncoderLayer
from Feature_extractors import FE


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class BERT(nn.Module):
	def __init__(self, mod, input_size, m, d_model, N, heads, dropout, custom_attn=True, multclass = False, NS_model=0):
		super(BERT, self).__init__()
		self.mod = mod
		self.multclass = multclass
		self.m = m
		self.encoder = Encoder(mod, d_model, N, heads, dropout, input_size, custom_attn, NS_model)
		if mod == "trx":
			d_model_reduced = int(d_model/4)
			#self.out = nn.Linear(d_model, 1)# This number can be changed
			self.out1 = nn.Linear(d_model, d_model_reduced)
			self.out2 = nn.Linear(d_model_reduced, 1)
		else:
			if multclass:
				self.out = nn.Linear(d_model, 2**m)
			else:
				self.out = nn.Linear(d_model, 2*m)
		self.dropout = nn.Dropout(dropout)
	def forward(self, src, mask, pe):
		enc_out = self.encoder(src, mask=mask, pe = pe)
		if self.mod == "rec":
			enc_out = self.out(enc_out)
		else:
			enc_out = self.out1(enc_out)
			enc_out = self.out2(enc_out)
		if self.mod == "rec":
			if self.multclass == False:
				batch = enc_out.size(0)
				ell = enc_out.size(1)
				enc_out = enc_out.contiguous().view(batch, ell*self.m,2)
				output = F.softmax(enc_out, dim=-1)
			else:
				output = F.softmax(enc_out, dim=-1)
		else:
			# encoders
			output = enc_out
		return output


class Encoder(nn.Module):
	def __init__(self, mod, d_model, N, heads, dropout, input_size, customAttn= True, NS_model = 0):
		super(Encoder, self).__init__()
		self.mod = mod
		self.Num_layers = N
		self.dropout = nn.Dropout(dropout)
		self.layers = get_clones(EncoderLayer(d_model, heads, dropout, custom_attn = customAttn), N)
		self.norm = nn.LayerNorm(d_model, eps=1e-5)
		self.NS_model = NS_model
		self.fe = FE(mod, NS_model, input_size, d_model)
	def forward(self, src, mask, pe):
		# input src.size  = [Batch, Seq_length, input_size]
		src = src.float()
		x = self.fe(src)
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
			self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True) # builtin version
		self.ffNet = FeedForward(d_model, d_ff=4 * d_model, dropout=dropout, act = 'relu')# linear -> act -> drop ->linear
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.custom = custom_attn

	def forward(self, x, mask):
		if self.norm_first: # This part is based on the order of the normalization
			# x.shape = [batchSize, seqLen, d_model]
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








