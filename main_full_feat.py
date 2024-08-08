import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
# from Feature_extractors import FE, ReducedFE
import copy
from parameters_full_distill import *
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
class FE_local(nn.Module):
	def __init__(self, mod, NS_model, input_size, d_model):
		super(FE_local, self).__init__()

		self.mod = mod
		self.NS_model = NS_model
		self.reserve = 3 + 8
		if self.NS_model == 3:
			self.FC1 = nn.Linear(input_size, d_model*2, bias=True)
			self.activation1 = nn.ReLU()
			self.FC2 = nn.Linear(d_model*2, d_model*2, bias=True)
			self.activation2 = nn.ReLU()
			self.FC3 = nn.Linear(d_model*2, d_model*2, bias=True)
			self.activation3 = nn.ReLU()
			self.FC4 = nn.Linear(d_model*4, d_model, bias=True)

	def forward(self, src):
		if self.NS_model == 3:

			x1 = self.FC1(src)
			x1_c = x1.clone()
			x1 = self.FC2(self.activation1(x1))
			x1 = self.FC3(self.activation2(x1))
			x = self.FC4(torch.cat([x1, -x1_c], dim = 2))

		return x

def ModelAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg

class ae_backbone(nn.Module):
	def __init__(self, arch, mod, input_size, m, d_model, dropout, custom_attn=True, multclass = False, NS_model=0):
		super(ae_backbone, self).__init__()
		self.arch = arch
		self.mod = mod
		self.multclass = multclass
		self.m = m

		if arch == "1xfe":
			self.fe1 = FE_local(mod, NS_model, input_size, d_model)
			self.norm1 = nn.LayerNorm(d_model, eps=1e-5)

   
		# project to the output space
		if mod == "trx":
			d_model_reduced = int(d_model/4)
			# self.out = nn.Linear(d_model, 1)# This number can be changed
			self.out1 = nn.Linear(d_model, d_model_reduced)
			self.out2 = nn.Linear(d_model_reduced, 1)
		else:
			if multclass:
				self.out = nn.Linear(d_model, 2**m)
			else:
				self.out = nn.Linear(d_model, 2*m)
		self.dropout = nn.Dropout(dropout)

	def forward(self, src, mask, pe):
		if self.arch == "1xfe" or "linear_enc_1xfe":
			enc_out = self.fe1(src)
			enc_out = self.norm1(enc_out)


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

########################## This is the overall AutoEncoder model ########################


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
		if args.embedding == True:
			self.Tmodel = ae_backbone(args.arch, "trx", args.clas+feature_dim, args.m, args.d_model_trx, args.dropout, args.custom_attn,args.multclass, args.enc_NS_model)
		else:
			self.Tmodel = ae_backbone(args.arch, "trx", args.m+feature_dim, args.m, args.d_model_trx, args.dropout, args.custom_attn,args.multclass, args.enc_NS_model)
		
		self.Rmodel = ae_backbone(args.arch, "rec", args.T, args.m, args.d_model_rec, args.dropout, args.custom_attn,args.multclass, args.dec_NS_model)
		if args.rev_iter > 0: ###################### Here we perform succesive refinement ######################################
			self.RmodelB = ae_backbone(args.arch, "rec", args.T+args.m, args.m, args.d_model_rec, args.dropout, args.custom_attn,args.multclass, args.dec_NS_model)
		########## Power Reallocation as in deepcode work ###############
		
		if self.args.reloc == 1:
			self.total_power_reloc = Power_reallocate(args)

	def power_constraint(self, inputs, isTraining, train_mean, train_std, idx = 0): # Normalize through batch dimension
		# this_mean = torch.mean(inputs, 0)
		# this_std  = torch.std(inputs, 0)
		if isTraining == 1 or train_mean is None:
			# training
			this_mean = torch.mean(inputs, 0)   
			this_std = torch.std(inputs, 0)
		elif isTraining == 0:
			# use stats from training
			this_mean = train_mean[idx]
			this_std = train_std[idx]

		outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
  
		return outputs, this_mean.detach(), this_std.detach()

	########### IMPORTANT ##################
	# We use unmodulated bits at encoder
	#######################################
	def forward(self, train_mean, train_std, bVec_md, fwd_noise_par, fb_noise_par, table = None, isTraining = 1):
		###############################################################################################################################################################
		combined_noise_par = fwd_noise_par + fb_noise_par # The total noise for parity bits
		for idx in range(self.args.T): # Go through T interactions
			# breakpoint()
			if self.args.features == "fpn":
				if idx == 0: # phase 0 
					src = torch.cat([bVec_md, torch.zeros(self.args.batchSize, self.args.ell, 2*(self.args.T-1)).to(self.args.device)],dim=2)
				elif idx == self.args.T-1:
					src = torch.cat([bVec_md, parity_all, combined_noise_par[:,:,:idx]],dim=2)
				else:
					src = torch.cat([bVec_md, parity_all, torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device),combined_noise_par[:,:,:idx],torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
			elif self.args.features == "fy":
				if idx == 0: # phase 0 
					src = torch.cat([bVec_md, torch.zeros(self.args.batchSize, self.args.ell, (self.args.T-1)).to(self.args.device)],dim=2)
				elif idx == self.args.T-1:
					src = torch.cat([bVec_md, parity_all + combined_noise_par[:,:,:idx]],dim=2)
				else:
					src = torch.cat([bVec_md, parity_all + combined_noise_par[:,:,:idx], torch.zeros(self.args.batchSize, args.ell, self.args.T-(idx+1) ).to(self.args.device)],dim=2)
			
			############# Generate the output ###################################################
			output = self.Tmodel(src, None, self.pe)
			parity, x_mean, x_std = self.power_constraint(output, isTraining, train_mean, train_std, idx)

			if self.args.reloc == 1:
				parity = self.total_power_reloc(parity,idx)
			if idx == 0:
				parity_fb = parity + combined_noise_par[:,:,idx].unsqueeze(-1)
				parity_all = parity
				received = parity + fwd_noise_par[:,:,0].unsqueeze(-1)
				x_mean_total, x_std_total = x_mean, x_std
			else:
				parity_fb = torch.cat([parity_fb, parity + combined_noise_par[:,:,idx].unsqueeze(-1)],dim=2) 
				parity_all = torch.cat([parity_all, parity], dim=2)     
				received = torch.cat([received, parity + fwd_noise_par[:,:,idx].unsqueeze(-1)], dim = 2)
				x_mean_total = torch.cat([x_mean_total, x_mean], dim = 0)
				x_std_total = torch.cat([x_std_total, x_std], dim = 0)
		# breakpoint()
		# ------------------------------------------------------------ receiver
		#print(received.shape)
		decSeq = self.Rmodel(received, None, self.pe) # Decode the sequence
		if args.rev_iter > 0:
			for i in range (args.rev_iter):
				if args.belief_modulate == True: # Modulate belief to align with the transmitted symbol power
					belief = 2*torch.matmul(decSeq, table)-1
				else:
					belief = torch.matmul(decSeq, table)
				received_wp = torch.cat([received,belief],dim=2)# received with prior
				decseq = self.RmodelB(received_wp, None, self.pe)
		
		return decSeq, x_mean_total, x_std_total

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
	pbar = tqdm(range(args.totalbatch))
	train_mean = torch.zeros(args.T, 1).to(args.device)
	train_std = torch.zeros(args.T, 1).to(args.device)
	for eachbatch in pbar:
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
		if args.snr2 >= 100:
			fb_noise_par = 0* fb_noise_par
		if np.mod(eachbatch, args.core) == 0:
			w_locals = []
			w0 = model.state_dict()
			w0 = copy.deepcopy(w0)
		else:
			# Use the common model to have a large batch strategy
			model.load_state_dict(w0)

		# feed into model to get predictions
		preds, batch_mean, batch_std = model(None, None, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)
		# breakpoint()
		if batch_mean is not None:
			train_mean += batch_mean
			train_std += batch_std # not the best way but numerically stable

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
		if eachbatch%5000 == 0:
			with torch.no_grad():
				# probs, decodeds = preds.max(dim=1)
				# succRate = sum(decodeds == ys.to(args.device)) / len(ys)
				print(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")
				logging.info(f"\nGBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")		
   
				# test with large batch every 1000 batches
				print("Testing started: ... ")
				logging.info("Testing started: ... ")
				# change batch size to 10x for testing
				args.batchSize = int(args.batchSize*10)
				EvaluateNets(model, None, None, args, logging)
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
			saveDir = f'{weights_folder}/model_weights' + str(eachbatch) + '.pt'
			torch.save(model.state_dict(), saveDir)
		pbar.update(1)
		pbar.set_description(f"GBAF train stats: batch#{eachbatch}, Loss {round(loss.item(), 8)}")

		# kill the training if the loss is nan
		if np.isnan(loss.item()):
			print("Loss is nan, killing the training")
			logging.info("Loss is nan, killing the training")
			break
  
	pbar.close()

	if train_mean is not None:
		train_mean = train_mean / args.totalbatch
		train_std = train_std / args.totalbatch	# not the best way but numerically stable
	  
	return train_mean, train_std

def EvaluateNets(model, train_mean, train_std, args, logging):
	if args.train == 0:
		
		path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
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
 
	start_time = time.time()
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
			preds, _, _ = model(train_mean, train_std, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)
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
				print(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}, num_pkts {num_pkts:.2e}")
				logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}, num_pkts {num_pkts:.2e}")
				print(f"Time elapsed: {(time.time() - start_time)/60} mins")
				logging.info(f"Time elapsed: {(time.time() - start_time)/60} mins")
			if args.train == 1:
				min_err = 20
			else:
				min_err = 100
			if symErrors > min_err or (args.train == 1 and num_batches_ran * args.batchSize * args.ell > 1e8):
				print(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}")
				logging.info(f"\nGBAF test stats: batch#{eachbatch}, SER {round(SER.item(), 10)}, numErr {symErrors.item()}")
				break
	# breakpoint()
	SER = symErrors.cpu() / (num_batches_ran * args.batchSize * args.ell)
	PER = pktErrors.cpu() / (num_batches_ran * args.batchSize)
	print(SER)
	print(f"Final test SER = {torch.mean(SER).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
	print(f"Final test PER = {torch.mean(PER).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
	logging.info(f"Final test SER = {torch.mean(SER).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
	logging.info(f"Final test PER = {torch.mean(PER).item()}, at SNR1 {args.snr1}, SNR2 {args.snr2} for rate {args.m}/{args.T}")
	# pdb.set_trace()


if __name__ == '__main__':
	# ======================================================= parse args
	args = args_parser()
	#args.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
	########### path for saving model checkpoints ################################
	args.saveDir = f'weights/model_weights{args.totalbatch-101}.pt'  # path to be saved to
	################## Model size part ###########################################
	args.d_model_trx = args.d_k_trx # total number of features
	args.d_model_rec = args.d_k_rec # total number of features
 
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
 
	parent_folder = f"jsac_results_final/{args.act}/N_{args.enc_NS_model}_{args.dec_NS_model}_d_{args.d_k_trx}_{args.d_k_rec}/snr2_{args.snr2}/seed_{args.seed}"
	# parent_folder = "temp"
 
	log_file = f"log_{sim_str}.txt"
	log_folder = f"{parent_folder}/logs/gbaf_{args.arch}_{args.features}/{folder_str}"
	log_file_name = os.path.join(log_folder, log_file)
 
	os.makedirs(log_folder, exist_ok=True)
	logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

	global weights_folder
	weights_folder = f"{parent_folder}/weights/gbaf_{args.arch}_{args.features}/{folder_str}/{sim_str}/"
	os.makedirs(weights_folder, exist_ok=True)


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
		print(f"Total number of trainable parameters: {num_params}")
		logging.info(f"Total number of trainable parameters: {num_params}")
  
		# print num params in Tmodel
		num_params = sum(p.numel() for p in model.Tmodel.parameters() if p.requires_grad)
		print(f"Total number of trainable parameters in Tmodel: {num_params}")
		logging.info(f"Total number of trainable parameters in Tmodel: {num_params}")
		# print num params in Rmodel
		num_params = sum(p.numel() for p in model.Rmodel.parameters() if p.requires_grad)
		print(f"Total number of trainable parameters in Rmodel: {num_params}")
		logging.info(f"Total number of trainable parameters in Rmodel: {num_params}")

		train_mean, train_std = train_model(model, args, logging)

		# stop training and test
		args.train = 0
		args.batchSize = int(args.batchSize*10)
		start_time = time.time()
  
		print("\nInference after training: ... ")
		logging.info("\nInference after training: ... ")
		EvaluateNets(model, None, None, args, logging)
		args.batchSize = int(args.batchSize/10)
  
		end_time = time.time()
		tot_time_mins = (end_time - start_time) / 60
		print(f"\nTime for testing: {tot_time_mins}")
		logging.info(f"\nTime for testing: {tot_time_mins}")

	## Inference
	print("\nInference using trained model and stats from large dataset: ... ")
	logging.info("\nInference using trained model and stats from large dataset: ... ")

	path = f'{weights_folder}/model_weights{args.totalbatch-101}.pt'
	print(f"\nUsing model from {path}")
	logging.info(f"\nUsing model from {path}")
 
	# use one very large batch to compute mean and std
	large_bs = int(1e6)
	args.batchSize = large_bs
	checkpoint = torch.load(path,map_location=args.device)

	# ======================================================= load weights
	model.load_state_dict(checkpoint)
	model = model.to(args.device)
	model.eval()
	map_vec = 2**(torch.arange(args.m))

	################################### Distance based vector embedding ####################
	A_blocks = torch.tensor([[0,0,0], [0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]],requires_grad=False).float() #Look up table for blocks
	Embed = torch.zeros(args.clas,large_bs, args.ell, args.clas)
	for i in range(args.clas):
		embed = torch.zeros(args.clas)
		for j in range(args.clas):
			if args.embed_normalize == True:
				embed[j] = (torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))-3/2)/0.866
			else:
				embed[j] = torch.sum(torch.abs(A_blocks[i,:]-A_blocks[j,:]))
		Embed[i,:,:,:]= embed.repeat(large_bs, args.ell, 1)

	if args.embedding == False:
		# BPSK modulated representations 
		bVec = torch.randint(0, 2, (large_bs, args.ell, args.m))
		bVec_md = 2*bVec-1
	else: # vector embedding
		bVec = torch.randint(0, args.clas, (large_bs, args.ell, 1))
		bVec_md = torch.zeros((large_bs, args.ell,args.clas), requires_grad=False) # generated data in terms of distance embeddings
		for i in range(args.clas):
			mask = (bVec == i).long()
			bVec_md= bVec_md + (mask * Embed[i,:,:,:])
	# generate n sequence
	std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
	std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
	fwd_noise_par = torch.normal(0, std=std1, size=(large_bs, args.ell, args.T), requires_grad=False)
	fb_noise_par = torch.normal(0, std=std2, size=(large_bs, args.ell, args.T), requires_grad=False)
	if args.snr2 == 100:
		fb_noise_par = 0* fb_noise_par

	inf_start_time = time.time()
	# feed into model to get predictions
	with torch.no_grad():
		preds, train_mean, train_std = model(None, None, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)
	inf_end_time = time.time()
	inf_time_secs = (inf_end_time - inf_start_time)
 
	# print the inference time
	print(f"Time to decode {large_bs} samples: {inf_time_secs} seconds")
  
	# breakpoint()
	# args.batchSize = int(args.batchSize*10)
	EvaluateNets(model, train_mean, train_std, args, logging)
