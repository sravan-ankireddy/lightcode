import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors import FE, ReducedFE
import copy
from parameters_full import *
import matplotlib.pyplot as plt

import logging
from tqdm import tqdm

##################### Author @Emre Ozfatura  @ Yulin Shao ###################################################

######################### Inlcluded modules and options #######################################
#1) Feature extracture 
#2) Successive decoding option 
#3) Vector embedding option 
#4) Belief Modulate

################################# Guideline #####################################
#Current activation is GELU
#trainining for 120000 epoch


################################## Distributed training approach #######################################################



def ModelAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg

class ae_backbone(nn.Module):
	def __init__(self, arch, mod, input_size, m, d_model, N, heads, dropout, custom_attn=True, multclass = False, NS_model=0):
		super(ae_backbone, self).__init__()
		self.arch = arch
		self.mod = mod
		self.multclass = multclass
		self.m = m
		# self.encoder = Encoder(mod, d_model, N, heads, dropout, input_size, custom_attn, attn_mode, pos_mode, layers_mode, NS_model)
		if mod == "trx":
			d_model_reduced = int(d_model/4)
			#self.out = nn.Linear(d_model, 1)# This number can be changed
			self.out1 = nn.Linear(d_model, d_model_reduced)
			self.out2 = nn.Linear(d_model_reduced, 1)
   
			# # add fully connected network
			# self.fc1 = nn.Linear(args.ell, 10*args.ell)
			# self.fc2 = nn.Linear(10*args.ell, args.ell)
		else:
			if multclass:
				# # add fully connected network
				# self.fc3 = nn.Linear(args.ell, 10*args.ell)
				# self.fc4 = nn.Linear(10*args.ell, args.ell)
				self.out = nn.Linear(d_model, 2**m)
			else:
				self.out = nn.Linear(d_model, 2*m)
		self.dropout = nn.Dropout(dropout)
  
		if arch == "1xfe":
			self.fe1 = FE(mod, NS_model, input_size, d_model)
			self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
		elif arch == "2xfe":
			self.fe1 = FE(mod, NS_model, input_size, d_model)
			self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
			self.fe2 = FE(mod, NS_model, d_model, d_model)
			self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
		elif arch == "3xfe":
			self.fe1 = FE(mod, NS_model, input_size, d_model)
			self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
			self.fe2 = FE(mod, NS_model, d_model, d_model)
			self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
			self.fe3 = FE(mod, NS_model, d_model, d_model)
			self.norm3 = nn.LayerNorm(d_model, eps=1e-5)
		elif arch == "rfe":
			self.rfe = ReducedFE(mod, NS_model, input_size, d_model)
			self.norm1 = nn.LayerNorm(d_model, eps=1e-5)

	def forward(self, src, mask, pe):
		if self.arch == "1xfe":
			enc_out = self.fe1(src)
			enc_out = self.norm1(enc_out)
		elif self.arch == "2xfe":
			enc_out = self.fe1(src)
			enc_out = self.norm1(enc_out)
			enc_out = self.fe2(enc_out)
			enc_out = self.norm2(enc_out)
		elif self.arch == "3xfe":
			enc_out = self.fe1(src)
			enc_out = self.norm1(enc_out)
			enc_out = self.fe2(enc_out)
			enc_out = self.norm2(enc_out)
			enc_out = self.fe3(enc_out)
			enc_out = self.norm3(enc_out)
		elif self.arch == "rfe":
			enc_out = self.rfe(src)
			enc_out = self.norm1(enc_out)

		# breakpoint()
		if self.mod == "rec":
			enc_out = self.out(enc_out)
		else:
			enc_out = self.out1(enc_out)
			enc_out = self.out2(enc_out)
   
			# # reverse the last two dimensions
			# enc_out = enc_out.permute(0,2,1)

			# enc_out = self.fc1(enc_out)
			# enc_out = self.fc2(enc_out)

			# # reverse the last two dimensions
			# enc_out = enc_out.permute(0,2,1)
		
		if self.mod == "rec":
			if self.multclass == False:
				batch = enc_out.size(0)
				ell = enc_out.size(1)
				enc_out = enc_out.contiguous().view(batch, ell*self.m,2)
				output = F.softmax(enc_out, dim=-1)
			else:
    
				# # reverse the last two dimensions
				# enc_out = enc_out.permute(0,2,1)

				# enc_out = self.fc3(enc_out)
				# enc_out = self.fc4(enc_out)
	
				# # reverse the last two dimensions
				# enc_out = enc_out.permute(0,2,1)
       
				output = F.softmax(enc_out, dim=-1)
		else:
			# encoders
			output = enc_out
		return output

########################## This is the overall AutoEncoder model ########################
def AWGNChan(X,SNR):
	sigma = np.sqrt(1/SNR)
	noise = sigma * np.random.randn(*X.shape)
	Y = X + noise
	return Y

def PAMmodulate(inbits, PAMSize):
	eta = np.sqrt(3 / ((2 ** (2*PAMSize)) - 1) )
	bitsyms = inbits.reshape([inbits.shape[0],-1,PAMSize])
	# create sym value array
	symval = 2**(np.arange(PAMSize,0,-1))
	X = np.matmul(bitsyms,symval)
	X_ref = (X/2).astype(int)

	# offset and normalization
	offset = 2**PAMSize - 1
	X = eta*(X-offset)

	return X, X_ref

def PAMdemodulate(Y,PAMSize,l):
	# offset correction and scaling
	offset = 2**PAMSize - 1
	eta = np.sqrt(3 / ((2 ** (2*PAMSize)) - 1) )
 
	Xh = np.round((Y/eta + offset)/2).astype(int)
	Xh = np.minimum(offset,np.maximum(0,Xh))
	
	Xb = np.zeros((Y.shape[0],Y.shape[1]*PAMSize))
	
	Xb = np.unpackbits(Xh.astype(np.uint8).reshape(-1, 1), axis=1)[:, -PAMSize:]
	Xb = Xb.reshape(Xh.shape[0], -1)

	return Xb, Xh

def SK_scheme(Theta,PAMSize,num_rounds,SNR):
	sigma2chan = 1/SNR
	sigma2n = 1/(1+SNR) # LMMSE estimate error
	alphan = 1/np.sqrt(sigma2n)
	
	# X = Theta
	X = Theta
	# print(Powercheck(X))
	Y = AWGNChan(X,SNR)
 
	Y_data = np.zeros((Theta.shape[0],Theta.shape[1],num_rounds))
	Y_data[:,:,0] = Y/(1+sigma2chan)

	K = Theta.shape[1]*PAMSize
	ThetaHat = Y/(1+sigma2chan)
	
	l = 3	
	Theta_hat = np.zeros((Theta.shape[0],Theta.shape[1],num_rounds))
	Theta_ind_hat = np.zeros((Theta.shape[0],Theta.shape[1],num_rounds))
	Theta_ind_hat_list = np.zeros((l,Theta.shape[0],Theta.shape[1],num_rounds))
	bits_hat = np.zeros((Theta.shape[0],K,num_rounds))

	Theta_hat[:,:,0] = ThetaHat
	
	bits_hat[:,:,0], Theta_ind_hat[:,:,0] = PAMdemodulate(ThetaHat,PAMSize,l)

	for i_r in range(1,num_rounds):
		errn = ThetaHat-Theta
  
		X = alphan*errn

		Y = AWGNChan(X,SNR)
		
		# advance
		betan = np.sqrt(sigma2n/sigma2chan) * np.sqrt(SNR)/(1+SNR)
		Y_data[:,:,i_r] = Y*betan
  
		ThetaHat = ThetaHat - Y*betan
		sigma2n = sigma2n/(1+SNR)
		alphan = 1/np.sqrt(sigma2n)
		
		# save data of all rounds
		Theta_hat[:,:,i_r] = ThetaHat
		
		bits_hat[:,:,i_r], Theta_ind_hat[:,:,i_r] = PAMdemodulate(ThetaHat,PAMSize,l)

	return Y_data#bits_hat[:,:,-1], Theta_hat, Theta_ind_hat, Theta_ind_hat_list, bits_hat, Theta_ind_hat[:,:,-1]

class AE(nn.Module):
	def __init__(self, args):
		super(AE, self).__init__()
		self.args = args
		################## We use learnable positional encoder which can be removed later ######################################
		#self.pe = PositionalEncoder(SeqLen=self.args.K+1, lenWord=args.d_model_trx) # learnable PE
		self.pe = PositionalEncoder_fixed()
  
		if self.args.features == "fpn":
			feature_dim = 2*(self.args.T-1)
		elif self.args.features == "fy":
			feature_dim = self.args.T-1

		########################################################################################################################
		# if args.embedding == True:
		# 	self.Tmodel = ae_backbone(args.arch, "trx", args.clas+feature_dim, args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
		# else:
		# 	self.Tmodel = ae_backbone(args.arch, "trx", args.m+feature_dim, args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
		
		self.Rmodel = ae_backbone(args.arch, "rec", args.T, args.m, args.d_model_trx, args.N_trx+1, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
		if args.rev_iter > 0: ###################### Here we perform succesive refinement ######################################
			self.RmodelB = ae_backbone(args.arch, "rec", args.T+args.m, args.m, args.d_model_trx, args.N_trx, args.heads_trx, args.dropout, args.custom_attn,args.multclass, args.NS_model)
		########## Power Reallocation as in deepcode work ###############
		
		if self.args.reloc == 1:
			self.total_power_reloc = Power_reallocate(args)

	def power_constraint(self, inputs, isTraining, eachbatch, idx = 0): # Normalize through batch dimension
		# this_mean = torch.mean(inputs, 0)
		# this_std  = torch.std(inputs, 0)
		if isTraining == 1:
			# training
			this_mean = torch.mean(inputs, 0)
			this_std  = torch.std(inputs, 0)
		elif isTraining == 0:
			# test
			if eachbatch == 0:
				this_mean = torch.mean(inputs, 0)
				this_std  = torch.std(inputs, 0)
				if not os.path.exists(stats_folder):
					os.mkdir(stats_folder)
				torch.save(this_mean, f'{stats_folder}/this_mean' + str(idx))
				torch.save(this_std, f'{stats_folder}/this_std' + str(idx))
				# print('this_mean and this_std saved ...')
			else:
				this_mean = torch.load(f'{stats_folder}/this_mean' + str(idx))
				this_std = torch.load(f'{stats_folder}/this_std' + str(idx))

		outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
		return outputs

	########### IMPORTANT ##################
	# We use unmodulated bits at encoder
	#######################################
	def forward(self, eachbatch, bVec_md, fwd_noise_par, fb_noise_par, table = None, isTraining = 1):
		###############################################################################################################################################################
		combined_noise_par = fwd_noise_par + fb_noise_par # The total noise for parity bits
  
		inbits = (bVec_md + 1)//2
		# convert to numpy
		inbits = inbits.cpu().numpy()
  
		Theta, ThetaInd = PAMmodulate(inbits,self.args.m)
  
		Theta = torch.from_numpy(Theta).float().to(self.args.device)
		ThetaInd = torch.from_numpy(ThetaInd).float().to(self.args.device)

		# SNR = 10**(self.args.snr1/10)
		# outbits, Theta_hat, Theta_ind_hat, Theta_ind_hat_list, bits_hat, Xh = SK_scheme(Theta,self.args.m,self.args.T,SNR)
		# Y_data = SK_scheme(Theta,self.args.m,self.args.T,SNR)
  
		# breakpoint()
		received = torch.zeros(self.args.batchSize, self.args.ell, (self.args.T)).to(self.args.device)
		decSeq = torch.zeros(self.args.batchSize, self.args.ell).to(self.args.device)
		for idx in range(self.args.T): # Go through T interactions

			if idx == 0: # phase 0 
				X = ThetaInd
			else:
				X = ThetaIndHat-ThetaInd
    
			# X = self.power_constraint(X, isTraining, eachbatch, idx)
			Y = X + combined_noise_par[:,:,idx]

			# received = torch.cat([Y, torch.zeros(self.args.batchSize, self.args.ell, (self.args.T-1)).to(self.args.device)],dim=2)
			received[:,:,idx] = Y
			decSeq = self.Rmodel(received, None, self.pe)
			# max along dim 2
			probs, ThetaIndHat = decSeq.max(dim=2)
			# breakpoint()

			# elif idx == self.args.T-1:
			# 	errn = ThetaIndHat-ThetaInd
  
			# 	X = self.power_constraint(X, isTraining, eachbatch, idx)
			# 	Y = X + combined_noise_par[:,:,idx].unsqueeze(-1)

			# 	received = torch.cat([Y, torch.zeros(self.args.batchSize, self.args.ell, (self.args.T-1)).to(self.args.device)],dim=2)
			# 	decSeq = self.Rmodel(received, None, self.pe)
			# 	# max along dim 2
			# 	probs, ThetaIndHat = decSeq.max(dim=2)
				
			# 	received = torch.cat([bVec_md, parity_all + combined_noise_par[:,:,:idx]],dim=2)
			# else:
			# 	received = torch.cat([bVec_md, parity_all + combined_noise_par[:,:,:idx], torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
			
			# ############# Generate the output ###################################################
			# output = self.Tmodel(src, None, self.pe)

			# parity = self.power_constraint(output, isTraining, eachbatch, idx) 
			# parity = self.total_power_reloc(parity,idx)
			# if idx == 0:
			# 	parity_fb = parity + combined_noise_par[:,:,idx].unsqueeze(-1)
			# 	parity_all = parity
			# 	received = parity + fwd_noise_par[:,:,0].unsqueeze(-1)
			# else:
			# 	parity_fb = torch.cat([parity_fb, parity + combined_noise_par[:,:,idx].unsqueeze(-1)],dim=2) 
			# 	parity_all = torch.cat([parity_all, parity], dim=2)     
			# 	received = torch.cat([received, parity + fwd_noise_par[:,:,idx].unsqueeze(-1)], dim = 2)
		# convert to torch
		# received = torch.from_numpy(Y_data).float().to(self.args.device)
		# # ------------------------------------------------------------ receiver
		# #print(received.shape)
		# decSeq = self.Rmodel(received, None, self.pe) # Decode the sequence
		if args.rev_iter > 0:
			for i in range (args.rev_iter):
				if args.belief_modulate == True: # Modulate belief to align with the transmitted symbol power
					belief = 2*torch.matmul(decSeq, table)-1
				else:
					belief = torch.matmul(decSeq, table)
				received_wp = torch.cat([received,belief],dim=2)# received with prior
				decseq = self.RmodelB(received_wp, None, self.pe)
		return decSeq




############################################################################################################################################################################








def train_model(model, args, logging):
	print("-->-->-->-->-->-->-->-->-->--> start training ...")
	logging.info("-->-->-->-->-->-->-->-->-->--> start training ...")
	model.train()
	start = time.time()
	epoch_loss_record = []
	flag = 0
	map_vec = 2**(torch.arange(args.m))
	################################### Distance based vector embedding ####################
	A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float() #Look up table for blocks
	Embed = torch.zeros(args.clas,args.batchSize, args.ell, args.clas)
	for i in range(args.clas):
		embed = torch.zeros(args.clas)
		for j in range(args.clas): ###### normalize vector embedding #########
			if args.embed_normalize == True:
				embed[j] = (torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))-3/2)/0.866 # normalize embedding
			else:
				embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
		Embed[i,:,:,:]= embed.repeat(args.batchSize, args.ell, 1)
	#########################################################################################
	pbar = tqdm(total=args.totalbatch)
	for eachbatch in range(args.totalbatch):
		if args.embedding == False:
			# BPSK modulated representations 
			bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
			bVec_md = 2*bVec-1
		else: # vector embedding
			bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
			bVec_md = torch.zeros((args.batchSize, args.ell,args.clas), requires_grad=False) # generated data in terms of distance embeddings
			for i in range(args.clas):
				mask = (bVec == i).long()
				bVec_md= bVec_md + (mask * Embed[i,:,:,:])
		#################################### Generate noise sequence ##################################################
		###############################################################################################################
		###############################################################################################################
		################################### Curriculum learning strategy ##############################################
		snr2=args.snr2
		if eachbatch < 0:#args.core * 30000:
			snr1=4* (1-eachbatch/(args.core * 30000))+ (eachbatch/(args.core * 30000)) * args.snr1
		else:
			snr1=args.snr1
		################################################################################################################
		std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
		std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
		# Noise values for the parity bits
		fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
		fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
		if args.snr2 == 100:
			fb_noise_par = 0* fb_noise_par
		if np.mod(eachbatch, args.core) == 0:
			w_locals = []
			w0 = model.state_dict()
			w0 = copy.deepcopy(w0)
		else:
			# Use the common model to have a large batch strategy
			model.load_state_dict(w0)

		# feed into model to get predictions
		preds = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)

		args.optimizer.zero_grad()
		if args.multclass:
			if args.embedding == False:
				bVec_mc = torch.matmul(bVec,map_vec)
				ys = bVec_mc.long().contiguous().view(-1)
			else:
				ys = bVec.contiguous().view(-1)
		else:
		# expand the labels (bVec) in a batch to a vector, each word in preds should be a 0-1 distribution
			ys = bVec.long().contiguous().view(-1)
		preds = preds.contiguous().view(-1, preds.size(-1)) #=> (Batch*K) x 2
		preds = torch.log(preds)
		# breakpoint()
		loss = F.nll_loss(preds, ys.to(args.device))########################## This should be binary cross-entropy loss
		loss.backward()
		####################### Gradient Clipping optional ###########################
		torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_th)
		##############################################################################
		args.optimizer.step()
		# Save the model
		w1 = model.state_dict()
		w_locals.append(copy.deepcopy(w1))
		###################### untill core number of iterations are completed ####################
		if np.mod(eachbatch, args.core) != args.core - 1:
			continue
		else:
			########### When core number of models are obtained #####################
			w2 = ModelAvg(w_locals) # Average the models
			model.load_state_dict(copy.deepcopy(w2))
			##################### change the learning rate ##########################
			if args.use_lr_schedule:
				args.scheduler.step()
		################################ Observe test accuracy ##############################
		if eachbatch%1000 == 0:
			with torch.no_grad():
				# probs, decodeds = preds.max(dim=1)
				# succRate = sum(decodeds == ys.to(args.device)) / len(ys)
				print(f"GBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")
				logging.info(f"GBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")		
   
				# test with large batch every 1000 batches
				print("Testing started: ... ")
				logging.info("Testing started: ... ")
				# change batch size to 10x for testing
				args.batchSize = int(args.batchSize*10)
				EvaluateNets(model, args, logging)
				args.batchSize = int(args.batchSize/10)
				print("... finished testing")
	
		####################################################################################
		if np.mod(eachbatch, args.core * 5000) == args.core - 1:
			epoch_loss_record.append(loss.item())
			if not os.path.exists(weights_folder):
				os.mkdir(weights_folder)
			torch.save(epoch_loss_record, f'{weights_folder}/loss')

		if np.mod(eachbatch, args.core * 5000) == args.core - 1:# and eachbatch >= 80000:
			if not os.path.exists(weights_folder):
				os.mkdir(weights_folder)
			saveDir = f'{weights_folder}/model_weights' + str(eachbatch)
			torch.save(model.state_dict(), saveDir)
		pbar.update(1)
		pbar.set_description(f"GBAF train stats: batch#{eachbatch}, Loss {round(loss.item(), 8)}")
	pbar.close()

def EvaluateNets(model, args, logging):
	if args.train == 0:
		
		path = f'{weights_folder}/model_weights120000'
		print(f"Using model from {path}")
		logging.info(f"Using model from {path}")
	
		checkpoint = torch.load(path,map_location=args.device)
	
		# ======================================================= load weights
		model.load_state_dict(checkpoint)
		model = model.to(args.device)
	model.eval()
	map_vec = 2**(torch.arange(args.m))

	args.numTestbatch = 100000000
	################################### Distance based vector embedding ####################
	A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float() #Look up table for blocks
	Embed = torch.zeros(args.clas,args.batchSize, args.ell, args.clas)
	for i in range(args.clas):
		embed = torch.zeros(args.clas)
		for j in range(args.clas):
			if args.embed_normalize == True:
				embed[j] = (torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))-3/2)/0.866
			else:
				embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
		Embed[i,:,:,:]= embed.repeat(args.batchSize, args.ell, 1)
	# failbits = torch.zeros(args.K).to(args.device)
	symErrors = 0
	pktErrors = 0
	for eachbatch in range(args.numTestbatch):
		if args.embedding == False:
			# BPSK modulated representations 
			bVec = torch.randint(0, 2, (args.batchSize, args.ell, args.m))
			bVec_md = 2*bVec-1
		else: # vector embedding
			bVec = torch.randint(0, args.clas, (args.batchSize, args.ell, 1))
			bVec_md = torch.zeros((args.batchSize, args.ell,args.clas), requires_grad=False) # generated data in terms of distance embeddings
			for i in range(args.clas):
				mask = (bVec == i).long()
				bVec_md= bVec_md + (mask * Embed[i,:,:,:])
		# generate n sequence
		std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
		std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
		fwd_noise_par = torch.normal(0, std=std1, size=(args.batchSize, args.ell, args.T), requires_grad=False)
		fb_noise_par = torch.normal(0, std=std2, size=(args.batchSize, args.ell, args.T), requires_grad=False)
		if args.snr2 == 100:
			fb_noise_par = 0* fb_noise_par

		# feed into model to get predictions
		with torch.no_grad():
			preds = model(eachbatch, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)
			# breakpoint()
			if args.multclass:
				if args.embedding == False:
					bVec_mc = torch.matmul(bVec,map_vec)
					ys = bVec_mc.long().contiguous().view(-1)
				else:
					ys = bVec.contiguous().view(-1)
			else:
				ys = bVec.long().contiguous().view(-1)
			preds1 =  preds.contiguous().view(-1, preds.size(-1))
			#print(preds1.shape)
			probs, decodeds = preds1.max(dim=1)
			decisions = decodeds != ys.to(args.device)
			symErrors += decisions.sum()
			SER = symErrors / (eachbatch + 1) / args.batchSize / args.ell
			pktErrors += decisions.view(args.batchSize, args.ell).sum(1).count_nonzero()
			PER = pktErrors / (eachbatch + 1) / args.batchSize
			
			num_batches_ran = eachbatch + 1
			num_pkts = num_batches_ran * args.batchSize	

			if eachbatch%1000 == 0:
				print(f"GBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}, num_pkts {num_pkts:.2e}")
				logging.info(f"GBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}, num_pkts {num_pkts:.2e}")

			if args.train == 1:
				min_err = 20
			else:
				min_err = 100
			if symErrors > min_err or (args.train == 1 and num_batches_ran * args.batchSize * args.ell > 1e8):
				print(f"GBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}")
				logging.info(f"GBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}")
				break
	# breakpoint()
	SER = symErrors.cpu() / (num_batches_ran * args.batchSize * args.ell)
	PER = pktErrors.cpu() / (num_batches_ran * args.batchSize)
	print(SER)
	print(f"Final test SER = {torch.mean(SER).item()}, at SNR {args.snr1} for rate {args.m}/{args.T}")
	print(f"Final test PER = {torch.mean(PER).item()}, at SNR {args.snr1} for rate {args.m}/{args.T}")
	logging.info(f"Final test SER = {torch.mean(SER).item()}, at SNR {args.snr1} for rate {args.m}/{args.T}")
	logging.info(f"Final test PER = {torch.mean(PER).item()}, at SNR {args.snr1} for rate {args.m}/{args.T}")
	# pdb.set_trace()


if __name__ == '__main__':
	# ======================================================= parse args
	args = args_parser()
	#args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
	########### path for saving model checkpoints ################################
	args.saveDir = 'weights/model_weights120000'  # path to be saved to
	################## Model size part ###########################################
	args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
 
	# fix the random seed for reproducibility
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	# torch.backends.cudnn.deterministic = True
	# torch.backends.cudnn.benchmark = False
 
 
	model = AE(args).to(args.device)
	# check if device contains the string 'cuda'
	if 'cuda' in args.device:
		# model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
		torch.backends.cudnn.benchmark = True
 
	# ======================================================= Initialize the model
	model = AE(args).to(args.device)
  
	# configure the logging
	folder_str = f"T_{args.T}/pow_{args.reloc}/{args.batchSize}/{args.lr}/"
	sim_str = f"K_{args.K}_m_{args.m}_snr1_{args.snr1}"
 
	parent_folder = f"results_temp/snr2_{args.snr2}/seed_{args.seed}"
 
	log_file = f"log_{sim_str}.txt"
	log_folder = f"{parent_folder}/logs/gbaf_{args.arch}_{args.features}/{folder_str}"
	log_file_name = os.path.join(log_folder, log_file)
 
	os.makedirs(log_folder, exist_ok=True)
	logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

	global weights_folder
	global stats_folder
	weights_folder = f"{parent_folder}/weights/gbaf_{args.arch}_{args.features}/{folder_str}/{sim_str}/"
	stats_folder = f"{parent_folder}/stats/gbaf_{args.arch}_{args.features}/{folder_str}/{sim_str}/"
	os.makedirs(weights_folder, exist_ok=True)
	os.makedirs(stats_folder, exist_ok=True)

	# ======================================================= run
	if args.train == 1:
		if args.opt_method == 'adamW':
			args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
		elif args.opt_method == 'lamb':
			args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
		else:
			args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
		if args.use_lr_schedule:
			lambda1 = lambda epoch: (1-epoch/args.totalbatch)
			args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
			######################## huggingface library ####################################################
			#args.scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=args.optimizer, warmup_steps=1000, num_training_steps=args.totalbatch, power=0.5)


		if 0:
			checkpoint = torch.load(args.saveDir)
			model.load_state_dict(checkpoint)
			print("================================ Successfully load the pretrained data!")

		# print the model summary
		print(model)
		logging.info(model)
  
		# print the number of parameters in the model that need to be trained
		num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
		print(f"Number of trainable parameters: {num_params}")
		logging.info(f"Number of trainable parameters: {num_params}")

		train_model(model, args, logging)
		# stop training and test
		args.train = 0
		args.batchSize = int(args.batchSize*10)
		EvaluateNets(model, args, logging)
	else:
		args.batchSize = int(args.batchSize*10)
		EvaluateNets(model, args, logging)