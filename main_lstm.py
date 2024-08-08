import torch, time, pdb, os, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils import *
from Feature_extractors import FE, ReducedFE
import copy
from parameters_lstm import *
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

enc_hidden_dim = 50
enc_num_layers = 1
dec_hidden_dim = 50
dec_num_layers = 1

def ModelAvg(w):
	w_avg = copy.deepcopy(w[0])
	for k in w_avg.keys():
		for i in range(1, len(w)):
			w_avg[k] += w[i][k]
		w_avg[k] = torch.div(w_avg[k], len(w))
	return w_avg

########################## This is the overall AutoEncoder model ########################


class AE(nn.Module):
	def __init__(self, args):
		super(AE, self).__init__()
		self.args = args

		# Encoder
		if args.embedding == True:
			self.Tmodel = torch.nn.LSTM(args.clas + 1, enc_hidden_dim, num_layers = enc_num_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
		else:
			self.Tmodel = torch.nn.LSTM(args.m + 1, enc_hidden_dim, num_layers = enc_num_layers, bias=True, batch_first=True, dropout=0, bidirectional=False)
		
		self.encoder_linear = torch.nn.Linear(enc_hidden_dim, 1)
		
		# Decoder
		self.Rmodel = torch.nn.LSTM(1, dec_hidden_dim, num_layers = dec_num_layers, bias=True, batch_first=True, dropout=0, bidirectional=True)
		self.decoder_linear = torch.nn.Linear(2*dec_hidden_dim, 2**args.m)
  
		########## Power Reallocation as in deepcode work ###############
		
		if self.args.reloc == 1:
			self.total_power_reloc = Power_reallocate(args)

	def power_constraint(self, inputs, isTraining, train_mean, train_std, eachbatch, idx = 0): # Normalize through batch dimension
		# this_mean = torch.mean(inputs, 0)
		# this_std  = torch.std(inputs, 0)

		if isTraining == 1 or train_std == []:
			# training
			this_mean = torch.mean(inputs, 0)
			this_std  = torch.std(inputs, 0)
		elif isTraining == 0:
			# use stats from training
			this_mean = train_mean[idx]
			this_std = train_std[idx]

		outputs = (inputs - this_mean)*1.0/ (this_std + 1e-8)
		return outputs, this_mean, this_std

	########### IMPORTANT ##################
	# We use unmodulated bits at encoder
	#######################################
	def forward(self, eachbatch, train_mean, train_std, bVec_md, fwd_noise_par, fb_noise_par, table = None, isTraining = 1):
		###############################################################################################################################################################
		combined_noise_par = fwd_noise_par + fb_noise_par # The total noise for parity bits
		for idx in range(self.args.T): # Go through T interactions
      
			if idx == 0: # 1st timestep
				input_total = torch.cat([bVec_md.view(self.args.batchSize, 1, self.args.K), torch.zeros((self.args.batchSize, 1, 1)).to(self.args.device)],dim=2)

				x_t_after_RNN, s_t_hidden  = self.Tmodel(input_total)
				x_t_tilde =   torch.selu(self.encoder_linear(x_t_after_RNN))
				
			else: # 2-30nd timestep
				input_total = torch.cat([bVec_md.view(self.args.batchSize, 1, self.args.K), z_t],dim=2)

				x_t_after_RNN, s_t_hidden  = self.Tmodel(input_total, s_t_hidden)
				x_t_tilde =   torch.selu(self.encoder_linear(x_t_after_RNN))

			############# Generate the output ###################################################

			x_t, x_mean, x_std = self.power_constraint(x_t_tilde, isTraining, train_mean, train_std, eachbatch, idx)
			if self.args.reloc == 1:
				x_t = self.total_power_reloc(x_t,idx)
    
			# check for unit power constraint on x_t
			# x_t_power = torch.mean(x_t**2)
			# fwd_noise_par_power_dB = 10*torch.log10(torch.mean(fwd_noise_par[:,:,idx]**2))
			# fb_noise_par_power_dB = 10*torch.log10(torch.mean(fb_noise_par[:,:,idx]**2))
			# print(f"Power of x_t: {x_t_power.item()}, fwd_noise_par_power_dB: {fwd_noise_par_power_dB.item()}, fb_noise_par_power_dB: {fb_noise_par_power_dB.item()}")
    
			# Forward transmission
			y_t = x_t + fwd_noise_par[:,:,idx].unsqueeze(-1)
			
			# Feedback transmission
			z_t = y_t + fb_noise_par[:,:,idx].unsqueeze(-1)
    
			# Concatenate values along time t
			if idx == 0:
				x_total = x_t
				y_total = y_t
				z_total = z_t
				x_mean_total, x_std_total = x_mean, x_std
			else:
				x_total = torch.cat([x_total, x_t ], dim = 1) # In the end, (batch, N, 1)
				y_total = torch.cat([y_total, y_t ], dim = 1) 
				z_total = torch.cat([z_total, z_t ], dim = 1)
				x_mean_total = torch.cat([x_mean_total, x_mean], dim = 0)
				x_std_total = torch.cat([x_std_total, x_std], dim = 0)
	
			# breakpoint()
		r_hidden, _  = self.Rmodel(y_total) # (batch, N, bi*hidden_size)
		output = torch.mean(self.decoder_linear(r_hidden), dim=1).unsqueeze(1) # (batch, 2^m)

		decSeq = F.softmax(output, dim=-1)
		# breakpoint()
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
		model.train()
  
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
		preds, batch_mean, batch_std = model(eachbatch, None, None, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)
		train_mean += batch_mean
		train_std += batch_std

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
				print(f"GBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")
				logging.info(f"GBAF train stats: batch#{eachbatch}, lr {args.lr}, snr1 {snr1}, snr2 {snr2}, BS {args.batchSize}, Loss {round(loss.item(), 8)}")		
   
				# test with large batch every 1000 batches
				print("Testing started: ... ")
				logging.info("Testing started: ... ")
				# change batch size to 10x for testing
				# args.batchSize = int(args.batchSize*10)
				train_mean = train_std = []
				EvaluateNets(model, train_mean, train_std, args, logging)
				# args.batchSize = int(args.batchSize/10)
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

	train_mean = train_mean / args.totalbatch
	train_std = train_std / args.totalbatch
	return train_mean, train_std

def EvaluateNets(model, train_mean, train_std, args, logging):
	if args.train == 0:
		
		path = f'{weights_folder}/model_weights{args.totalbatch-101}'
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
			preds, _, _ = model(eachbatch, train_mean, train_std, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=0)
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
 
	parent_folder = f"results_lstm/snr2_{args.snr2}/seed_{args.seed}"
 
	# parent_folder = "temp"
 
	log_file = f"log_{sim_str}.txt"
	log_folder = f"{parent_folder}/logs/lstm/enc_{enc_num_layers}_{enc_hidden_dim}_dec_{dec_num_layers}_{dec_hidden_dim}/{folder_str}"
	log_file_name = os.path.join(log_folder, log_file)
 
	os.makedirs(log_folder, exist_ok=True)
	logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

	global weights_folder
	global stats_folder
	weights_folder = f"{parent_folder}/weights/lstm/enc_{enc_num_layers}_{enc_hidden_dim}_dec_{dec_num_layers}_{dec_hidden_dim}/{folder_str}/{sim_str}/"
	stats_folder = f"{parent_folder}/stats/lstm/enc_{enc_num_layers}_{enc_hidden_dim}_dec_{dec_num_layers}_{dec_hidden_dim}/{folder_str}/{sim_str}/"
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
		model.train()
		train_mean, train_std = train_model(model, args, logging)
		# stop training and test
		args.train = 0

		# args.batchSize = int(args.batchSize*10)
		EvaluateNets(model, train_mean, train_std, args, logging)
	else:
		args.batchSize = int(args.batchSize*4)
  
		# precompute mean and std for the rntire training set
		num_train_samps = int(1e8)
		num_batches = int(num_train_samps/args.batchSize)

		path = f'{weights_folder}/model_weights120000'
		print(f"Using model from {path}")
		logging.info(f"Using model from {path}")

		checkpoint = torch.load(path,map_location=args.device)

		# ======================================================= load weights
		model.load_state_dict(checkpoint)
		model = model.to(args.device)
		model.eval()
		map_vec = 2**(torch.arange(args.m))

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
  
		train_mean = torch.zeros(args.T, 1).to(args.device)
		train_std = torch.zeros(args.T, 1).to(args.device)
		for eachbatch in tqdm(range(num_batches)):
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
				preds, x_mean, x_std = model(eachbatch, train_mean, train_std, bVec_md.to(args.device), fwd_noise_par.to(args.device), fb_noise_par.to(args.device), A_blocks.to(args.device), isTraining=1)
			# breakpoint()
			train_mean += x_mean
			train_std += x_std

		train_mean = train_mean / num_batches
		train_std = train_std / num_batches
		# breakpoint()
		EvaluateNets(model, train_mean, train_std, args, logging)
