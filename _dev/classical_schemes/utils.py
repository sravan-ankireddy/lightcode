import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import random
import torch.nn.functional as F
import warnings
from tqdm import tqdm
from tqdm import trange
import time
import scipy
import scipy.special
import matplotlib.pyplot as plt
from generate_sk import AWGNChan, Powercheck, generate_sk_data

def Powercheck(x):
	return (torch.mean(torch.abs(x)**2))

def qfunction(z):
	return(0.5-0.5*scipy.special.erf(z/np.sqrt(2)))

def dec2bin(x, bits):
	# mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
	mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
	return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

def const_wt_data(wt,block_len,batch_size,device):
	if (wt == 0):
		data_0 = np.zeros((batch_size,block_len))
		data = torch.from_numpy(data_0).to(device).float()
	elif (wt == 1):
		num_rep_w1 = int(batch_size/block_len)+1
		data = np.tile(np.eye(block_len),(num_rep_w1,1))         
		data = torch.from_numpy(data[:batch_size,:]).to(device).float()
	elif (wt == 2):
		num_w2 = int(block_len*(block_len-1)/2)
		num_rep_w2 = int(batch_size/num_w2)+1

		data_2 = np.zeros((num_w2,block_len))
		count = 0
		for ii in range(block_len):
			for jj in range(ii+1,block_len):
				data_2[count,ii] = 1
				data_2[count,jj] = 1
				count = count + 1
		data = np.tile(data_2,(num_rep_w2,1))         
		data = torch.from_numpy(data[:batch_size,:]).to(device).float()
  
	elif (wt == 3):
		num_w3 = int(block_len*(block_len-1)*(block_len-2)/6)
		num_rep_w3 = int(batch_size/num_w3)+1

		data_3 = np.zeros((num_w3,block_len))
		count = 0
		for ii in range(block_len):
			for jj in range(ii+1,block_len):
				for kk in range(jj+1,block_len):
					data_3[count,ii] = 1
					data_3[count,jj] = 1
					data_3[count,kk] = 1
					count = count + 1

		data = np.tile(data_3,(num_rep_w3,1))         
		data = torch.from_numpy(data[:batch_size,:]).to(device).float()
  
	return data

def powerblastErf(P,spa,sigman,list_decoding):
	"""
	Compute the probability of error if a power blast scheme is to be applied
	The actual decoder is not implemented
	Only the error is directly computed
	"""
	# ref code
	# A = np.sqrt(P/spa)
	# sigmasqn = sigman**2
	# gamma = (A/2) + (sigmasqn/A)*np.log((1-spa)/spa)
	
	# Pe1 = qfunction(gamma/sigman)
	# Pe2 = qfunction((A-gamma)/sigman)

	# Perr = (1-spa)*Pe1 + spa*Pe2
	###########################################################
	if list_decoding:
		c = np.sqrt((1-spa)/spa)
		A = c - 1/c
		B = c + 1/c

		sigmasqn = sigman**2
		gamma = (A/2) + (sigmasqn/B)*np.log((1-spa)/spa)
		
		Pe_0_to_1 = qfunction((gamma+1/c)/sigman)
		Pe_1_to_0 = 1 - qfunction((gamma-c)/sigman)
  
		Perr = (1-spa)*Pe_0_to_1 + spa*Pe_1_to_0
  
	else: # FIX ME
		sigmasqn = sigman**2  
  
		c = np.sqrt(1/spa)
		gamma = c/2 + (sigmasqn/c)*np.log(2*(1-spa)/spa)
		
		Pe_0_to_1 = 2*qfunction(gamma/sigman)
		Pe_1_to_0 = qfunction((c-gamma)/sigman)
  
		Perr = (1-spa)*Pe_0_to_1 + spa*Pe_1_to_0

	return Perr


def powerBlastBer_real(logging, input_data_ref, target_data, data_type, pam_sym, prob_test, batch_size, num_test_batches, compute_err_stats, snr_test, device, list_decoding):

	ser = 0

	block_len = input_data_ref.size()[1]
	max_blocks = input_data_ref.size()[0]

	sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
	num_err = 0
  
	# Error Stats:
	num_err_per_cw_wt = np.zeros((1,block_len))	
	decoded_weights = np.zeros((1,block_len)) 
	input_cw_count = np.zeros((1,block_len))	
 
	min_errors = 2e2
	batch_count = 0
	num_err = 0
	cur_idx = 0
 
	pbar = trange(num_test_batches, desc='Track num. err')
	for i_ib in pbar:
	   
		# generate new data each for each mninibatch if iid
		if (data_type == "sk_iid" or data_type == "sk_iid_2" or data_type == "sk_iid_3" or data_type == "sk_iid_4" or data_type == "sk_iid_5" or data_type == "cs_iid"):
			input_data = torch.tensor(np.random.choice(pam_sym,(batch_size,block_len),p=prob_test)).to(device).float()
		# read from offline data 
		else:	
			cur_idx = cur_idx % max_blocks
			input_data = input_data_ref[cur_idx:cur_idx+batch_size,:]
	 		
		# shift to make mean 0
		data = (input_data - torch.mean(input_data)).to(device)

		# scale such that expected power of symbol is 1
		pow_norm = torch.linalg.matrix_norm(data)/np.sqrt(block_len*batch_size) 
		x = data/pow_norm
  
		noise = sigma * torch.randn(x.shape[0],x.shape[1]).to(device)

		y = x + noise
  
		if list_decoding:
			# decode using MAP threshold rule : list decoding available
			spa = torch.mean(torch.abs(input_data))
			c = pow_norm/spa
			A = c - 1/c
			B = c + 1/c

			sigmasqn = sigma**2
			gamma = (A/2) + (sigmasqn/B)*torch.log((1-spa)/spa)
			# FIX ME
			gamma_2 = (2-torch.mean(input_data))/pow_norm - gamma

			input_data_hat = torch.zeros_like(y)
			input_data_hat[y > gamma] = 1
			input_data_hat[y > gamma_2] = 2

			# check for total no. errors
			num_err = num_err + torch.sum(input_data_hat != input_data)
	
			# err_vec 
			est = input_data_hat.cpu().numpy()
			lab = input_data.cpu().numpy()

		else:
			# decode using MAP threshold rule : list decoding available
			spa = torch.mean(torch.abs(input_data))

			sigmasqn = sigma**2
	
			c = torch.sqrt(1/spa)
			gamma = c/2 + (sigmasqn/c)*torch.log(2*(1-spa)/spa)

			input_data_hat = torch.zeros_like(y)
			input_data_hat[y < -gamma] = -1
			input_data_hat[y > gamma] = 1

			# check for total no. errors
			num_err = num_err + torch.sum(input_data_hat != input_data)
	
			# err_vec 
			est = input_data_hat.cpu().numpy()
			lab = input_data.cpu().numpy()
   
		err_vec = (est != lab)

		if compute_err_stats:  
			# update error stats
			lab = np.abs(lab)
			ref_ind = np.sum(lab, axis=1).astype(int)

			input_cw_count += np.bincount(ref_ind, minlength=input_cw_count.size)
	
			for ie in range(block_len):
				# get list of indices where the input weight is ie
				ind = np.where(ref_ind == ie)[0]
				if (ind.size > 0):
					num_err_per_cw_wt[0,ie] += np.sum(err_vec[ind,:])
					decoded_weights[0,ie] += np.sum(est[ind,:])

		cur_idx += batch_size
		batch_count += 1

		cur_ber = num_err/(block_len*batch_size*batch_count)
		pbar.set_description(f'PB: Num err. {num_err} -> {int(min_errors)}, current ber {cur_ber}')
  
		# break if min errors are reached
		if (num_err > min_errors):
			break
	
	err_rate = num_err/(block_len*batch_size*batch_count)
	ser = err_rate

	if compute_err_stats:
		mean_err_weight_per_input = np.divide(num_err_per_cw_wt, input_cw_count*block_len)
		mean_err_weight_per_input[np.isnan(mean_err_weight_per_input)] = 0
	
		err_fraction_per_wt = np.divide(num_err_per_cw_wt,	np.sum(num_err_per_cw_wt))
	
		weighted_err_frac = np.multiply(err_fraction_per_wt, input_cw_count/np.sum(input_cw_count))
		weighted_err_frac[np.isnan(weighted_err_frac)] = 0

	# Print the necessary statistics
	
	print("Emp Power blast: SNR : ", snr_test, ", SER:", float(ser), " ran a total of ", int(batch_size*batch_count), " blocks ", int(block_len*batch_size*batch_count), " symbols")
	logging.info("\nEmp Power blast: SNR : {}, SER: {}, ran a total of {} blocks , {} symbols".format(snr_test, float(ser), int(batch_size*batch_count), int(block_len*batch_size*batch_count)))
 
	if compute_err_stats:
		print("Emp Power blast: Avg. SER per input wt category: ", mean_err_weight_per_input)
		logging.info("\nEmp Power blast: Avg. SER per input wt category: {}".format(mean_err_weight_per_input))
	
		print("Emp Power blast: Input codeword freq. per weight - percentage: ", input_cw_count*100/np.sum(input_cw_count))
		logging.info("\nEmp Power blast: Input codeword freq. per weight - percentage: {}".format(input_cw_count*100/np.sum(input_cw_count)))

		print("Emp Power blast: Percentage of errors per wt category: ", weighted_err_frac*100/np.sum(weighted_err_frac))
		logging.info("\nEmp Power blast: Percentage of errors per wt category: {}".format(weighted_err_frac*100/np.sum(weighted_err_frac)))

	return ser

# create a function to calculate the moving average using convolution
def moving_avg(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

def train_enc_dec_real(logging, fine_tune, epoch, data_type, prob_train, model_enc, model_dec, model_mode, model_type, alt_train, pam_sym, batch_size, grad_acc_step, block_len, device, optimizer_enc, optimizer_dec, train_snr, input_data, target_data, norm_mode = "train", train_mean=0, train_std=1):
	if (alt_train):
		if (model_type == "enc"):
			model_enc.train()
			model_dec.eval()
		else:
			model_enc.eval()
			model_dec.train()
	else:
		if (model_mode == "enc"):
			model_enc.train()
		elif (model_mode == "dec"):
			model_dec.train()
		elif (model_mode == "enc_dec"):
			model_enc.train()
			model_dec.train()
	
	# turn off batch normalization  across all layers in fine_tune mode and  use precomputed mean and std
	if (fine_tune):
		for m in model_enc.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.eval()
				m.track_running_stats = False
		for m in model_dec.modules():
			if isinstance(m, nn.BatchNorm1d):
				m.eval()
				m.track_running_stats = False
    
		# override the prob_train
		prob_train_new = prob_train.copy()
		prob_train_new[2] = 0.05
		prob_train_new = prob_train_new/np.sum(prob_train_new)

	train_loss = 0
	num_train_batch = 1000
	
	# use gradient accumulation only at the end when batch_size is very large
	if (batch_size <= 100000):
		grad_acc_step = 1

	if (grad_acc_step > 1):
		num_train_batch = int(num_train_batch*grad_acc_step)

	cur_idx = 0
	max_blocks = input_data.size()[0]
 
	pam_size = len(pam_sym)
	if (pam_size == 2):
		criterion = nn.BCEWithLogitsLoss()
		loss_type = "BCE"
	else:
		# class_weights = 1.0 / np.array(prob_train)
		# class_weights = class_weights / class_weights.sum()
		# criterion = nn.CrossEntropyLoss(weight=class_weights)
		criterion = nn.CrossEntropyLoss()
		loss_type = "CE"
	train_snr_ref = train_snr
 
	random_snr = 0
  
	tot_loss = 0
	for batch_idx in tqdm(range(num_train_batch)): 
		# generate new data each for each mninibatch if iid
		data = torch.tensor(np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)).to(device).float()
		labels = data.clone().detach()
		# reads from offline data 
		if ("sk_real" in data_type):	
			cur_idx = cur_idx % max_blocks
			data = input_data[cur_idx:cur_idx+batch_size,:]
			labels = target_data[cur_idx:cur_idx+batch_size,:]
		
		# pick the snr from randomly from a range when training the decoder: inspired by ProductAE
		if random_snr:
			train_snr = np.random.uniform(train_snr_ref-1,train_snr_ref+1,data.shape)
			sigma = torch.sqrt(torch.tensor(10**(-train_snr/10))).to(device)	
			noise = (sigma * torch.randn(data.shape).to(device)).float()	
		else:
			sigma = torch.sqrt(torch.tensor(10**(-train_snr/10)))
			noise = sigma * torch.randn(data.shape).to(device)

		x, temp_out, temp_out = model_enc(data, norm_mode, train_mean, train_std)

		y = x + noise
  
		estimate = model_dec(y)

		# reshape estimate and make labels non negative
		if (pam_size > 2):
			estimate = torch.permute(estimate, (0,2,1))
			labels[labels == -1] = 2
			loss = criterion(estimate, labels.long())
		else:
			loss = criterion(estimate, labels)

		# update tot_loss
		tot_loss += loss.item()
  
		# normalize loss to account for batch accumulation
		loss = loss / grad_acc_step 

		loss.backward()

		train_loss += loss.item()
    
		# gradient accumulation step
		if ((batch_idx+1) % grad_acc_step == 0):
			if (alt_train):
				if (model_type == "enc"):
					optimizer_enc.step()
					optimizer_enc.zero_grad()
				else:
					optimizer_dec.step()
					optimizer_dec.zero_grad()
			else:
				if (model_mode == "enc"):
					optimizer_enc.step()
					optimizer_enc.zero_grad()
				elif (model_mode == "dec"):
					optimizer_dec.step()
					optimizer_dec.zero_grad()
				elif (model_mode == "enc_dec"):
					optimizer_enc.step()
					optimizer_dec.step()
					optimizer_enc.zero_grad()
					optimizer_dec.zero_grad()
			
		cur_idx += batch_size
			
		# scheduler_enc.step(epoch + batch_idx/num_train_batch)
		# scheduler_dec.step(epoch + batch_idx/num_train_batch)
	train_snr = np.mean(train_snr)
	if (alt_train):
		if (model_type == "enc"):
			print('====> Enc trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type, epoch, train_snr, train_loss /num_train_batch))
			logging.info('====> Enc trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type,epoch, train_snr, train_loss /num_train_batch))
		else:
			print('====> Dec trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type, epoch, train_snr, train_loss /num_train_batch))
			logging.info('====> Dec trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type, epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> (Enc,Dec) trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type, epoch, train_snr, train_loss /num_train_batch))
		logging.info('====> (Enc,Dec) trained: Loss type: {}, Epoch: {}, SNR: {}, Average loss: {:.8f}'.format(loss_type, epoch, train_snr, train_loss /num_train_batch))
  
	mean_loss = tot_loss/num_train_batch
	
	return mean_loss


def test_enc_dec_real(logging, data_type, prob_test, model_enc, model_dec, model_mode, pam_sym, batch_size, num_test_batches, compute_err_stats, block_len, device, optimizer_enc, optimizer_dec, test_snr, input_data, target_data, norm_mode = "train", train_mean=0, train_std=1):
	model_enc.eval()
	model_dec.eval()
	test_ber=.0

	snr_test = test_snr

	cur_idx = 0
	max_blocks = input_data.size()[0]
	min_errors = 2e2
 
	pam_size = len(pam_sym)
 
	# Error Stats: 
	num_err_per_cw_wt = np.zeros((1,block_len))	
	decoded_weights_hist = np.zeros((1,block_len))
	input_cw_count = np.zeros((1,block_len))
 
	pbar = trange(num_test_batches, desc='Track num. err')
 
	with torch.no_grad():
		batch_count = 0
		num_err = 0
		for batch_idx in pbar:
   
			# generate new data each for each mninibatch if iid
			data = torch.tensor(np.random.choice(pam_sym,(batch_size,block_len),p=prob_test)).to(device).float()
			labels = data.clone().detach()
			# reads from offline data 
			if ("sk_real" in data_type):	
				cur_idx = cur_idx % max_blocks
				data = input_data[cur_idx:cur_idx+batch_size,:]
				labels = target_data[cur_idx:cur_idx+batch_size,:]

			if 0:
				# FIX ME : generate data every mini batch
				PAMSize = 3
				num_rounds_sk = 8
				num_batches = 100
				list_decoding = 1
				list_size = 3
				K = int(block_len*PAMSize)

				sk_data_train_snr = generate_sk_data(K,2**PAMSize,num_rounds_sk,1,batch_size,snr_test)
    
				Theta_ind_data = np.squeeze(sk_data_train_snr[3])
				Theta_ind_hat_list = np.squeeze(sk_data_train_snr[6][:,:,:,:,-2])
				pos_vec = np.full((Theta_ind_data.shape[0],Theta_ind_data.shape[1]), list_size)

				if list_decoding == 1:         
					for i in range(list_size):
						# create a boolean mask indicating where the elements match
						eq_mask = (Theta_ind_data == Theta_ind_hat_list[i,:,:])
						
						# update pos only if previous val was list_size
						update_mask = (pos_vec == list_size)
						
						mask = np.logical_and(eq_mask,update_mask)

						# update the corresponding elements of pos to the matching indices
						pos_vec[mask] = i
      
				data = torch.tensor(pos_vec.T).to(device).float()
				labels = data.clone().detach()

			sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
			noise = sigma * torch.randn(data.shape).to(device)

			x, temp_mean, temp_std = model_enc(data, norm_mode, train_mean, train_std) 

			y = x + noise
			# breakpoint()
			estimate = model_dec(y)

			if (pam_size > 2):
				# apply logsoftmax
				estimate = torch.log_softmax(estimate,dim=2)
				
				# reshape estimate and make labels non negative
				labels[labels == -1] = 2
				estimate = torch.argmax(estimate,dim=2)
			else:
				estimate = torch.sigmoid(estimate)
				estimate = torch.round(estimate)

			# err_vec 
			est = estimate.cpu().numpy()
			lab = labels.cpu().numpy()
			err_vec = (est != lab)
			test_ber += np.mean(err_vec)
   
			cur_idx += batch_size
   
			batch_count += 1
			num_err += np.sum(err_vec)

			if compute_err_stats:
				# update error stats
				lab = np.abs(lab)
				est = np.abs(est)
    
				# get labels with 2's 
				lab_2 = (lab == 2)
				ref_ind = np.sum(lab_2, axis=1).astype(int)

				input_cw_count += np.bincount(ref_ind, minlength=input_cw_count.size)
			
				for ie in range(block_len):
					# get list of indices where the input weight is ie
					ind = np.where(ref_ind == ie)[0]
					if (ind.size > 0):
						num_err_per_cw_wt[0,ie] += np.sum(err_vec[ind,:])

			cur_ber = test_ber/batch_count
			pbar.set_description(f'SC: Num err. {num_err} -> {int(min_errors)}, current ber {cur_ber}')
			if (num_err > min_errors):
				break

		# plot the power of input and enc
		data_pb = (data - torch.mean(data))/torch.std(data)
		data_pb_norm = torch.norm(data_pb,dim=1).detach().cpu().numpy()
		data_wt = torch.sum(data,dim=1).detach().cpu().numpy()
		x_norm = torch.norm(x,dim=1).detach().cpu().numpy()
		plot_ind = np.argsort(data_wt)

		plt.figure()
		leg = []
		plt.plot(data_wt[plot_ind])
		leg_str = f'Input codeword weight'
		leg.append(leg_str)
		plt.plot(x_norm[plot_ind])
		leg_str = f'Norm of encoded codeword'
		leg.append(leg_str)
  
		# plot moving avg of power for encoded codeword
		x_norm_avg = np.convolve(x_norm[plot_ind], np.ones((100,))/100, mode='valid')
		plt.plot(x_norm_avg)
		leg_str = f'Norm of encoded codeword (moving avg, window=100)'
		leg.append(leg_str)
  
		plt.plot(data_pb_norm[plot_ind])
		leg_str = f'Norm of power blast codeword'
		leg.append(leg_str)
		plt.legend(leg)
		plt.show()
		# save plot
		spa_val = np.round(1 - prob_test[1],3)
		plt.savefig(f'temp_power_plots/power_plot_snr_{model_mode}_{data_type}_snr_{test_snr}_spa_{spa_val}.png')
		
		# breakpoint()
		test_ber  /= 1.0*batch_count

	err_rate = num_err/(block_len*batch_size*batch_count)
	ser = err_rate
 
	if compute_err_stats:
		mean_err_weight_per_input = np.divide(num_err_per_cw_wt, input_cw_count*block_len)
		mean_err_weight_per_input[np.isnan(mean_err_weight_per_input)] = 0
	
		err_fraction_per_wt = np.divide(num_err_per_cw_wt,	np.sum(num_err_per_cw_wt))
	
		weighted_err_frac = np.multiply(err_fraction_per_wt, input_cw_count/np.sum(input_cw_count))
		weighted_err_frac[np.isnan(weighted_err_frac)] = 0

	# Print the necessary statistics
	print("SNR : ", snr_test, ", SER:", float(test_ber), " ran a total of ", int(batch_size*batch_count), " blocks ", int(block_len*batch_size*batch_count), " symbols")
	logging.info("\nSNR : {}, SER: {}, ran a total of {} blocks , {} symbols".format(snr_test, float(ser), int(batch_size*batch_count), int(block_len*batch_size*batch_count)))
 
	if compute_err_stats: 
		print("Avg. SER per input wt category: ", mean_err_weight_per_input)
		logging.info("\nAvg. SER per input wt category: {}".format(mean_err_weight_per_input))
	
		print("Input codeword freq. per weight - percentage:: ", input_cw_count*100/np.sum(input_cw_count))
		logging.info("\nInput codeword freq. per weight - percentage: {}".format(input_cw_count*100/np.sum(input_cw_count)))

		print("Percentage of errors per wt category: ", weighted_err_frac*100/np.sum(weighted_err_frac))
		logging.info("\nPercentage of errors per wt category: {}".format(weighted_err_frac*100/np.sum(weighted_err_frac)))

	return float(test_ber)














####### qam models #######

def train_enc_dec_real_qam(epoch, model_enc, model_dec, model_mode, model_type, alt_train, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, train_snr, input_data, target_data):
	if (alt_train):
		if (model_type == "enc"):
			model_enc.train()
			model_dec.eval()
		else:
			model_enc.eval()
			model_dec.train()
	else:
		if (model_mode == "enc"):
			model_enc.train()
		elif (model_mode == "dec"):
			model_dec.train()
		elif (model_mode == "enc_dec"):
			model_enc.train()
			model_dec.train()

	train_loss = 0
	num_train_batch = 1000

	cur_idx = 0
	max_blocks = input_data.size()[0]
	for batch_idx in tqdm(range(num_train_batch)):
		
		# FIX ME
		cur_idx = cur_idx % max_blocks
		data = input_data[cur_idx:cur_idx+batch_size,:]
		labels = target_data[cur_idx:cur_idx+batch_size,:]
		
		sigma = torch.sqrt(torch.tensor(10**(-train_snr/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		if (alt_train):
			if (model_type == "enc"):
				optimizer_enc.zero_grad()
			else:
				optimizer_dec.zero_grad()
		else:
			if (model_mode == "enc"):
				optimizer_enc.zero_grad()
			elif (model_mode == "dec"):
				optimizer_dec.zero_grad()
			elif (model_mode == "enc_dec"):
				optimizer_enc.zero_grad()
				optimizer_dec.zero_grad()

		y = model_enc(data, noise)

		estimate = model_dec(y)
  
		####### FIX ME #######
		# estimate = estimate.view(-1,8)
		# labels = labels.reshape(-1).long().to(device)
		estimate = torch.permute(estimate, (0,2,1))
		# breakpoint()

		loss = F.nll_loss(estimate, labels.long())
		# labels_bin = dec2bin(labels,3)
		# breakpoint()
		# loss = F.binary_cross_entropy_with_logits(estimate, labels_bin)

		loss.backward()
		train_loss += loss.item()
		if (alt_train):
			if (model_type == "enc"):
				optimizer_enc.step()
			else:
				optimizer_dec.step()
		else:
			if (model_mode == "enc"):
				optimizer_enc.step()
			elif (model_mode == "dec"):
				optimizer_dec.step()
			elif (model_mode == "enc_dec"):
				# clip_value = 0.5
				# nn.utils.clip_grad_value_(model_enc.parameters(), clip_value)
				# nn.utils.clip_grad_value_(model_dec.parameters(), clip_value)
				
				optimizer_enc.step()
				optimizer_dec.step()
			
		cur_idx += batch_size
			
		# scheduler_enc.step(epoch + batch_idx/num_train_batch)
		# scheduler_dec.step(epoch + batch_idx/num_train_batch)
	if (alt_train):
		if (model_type == "enc"):
			print('====> Enc trained Epoch: {}, SNR: {}, Average NLL loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
		else:
			print('====> Dec trained Epoch: {}, SNR: {}, Average NLL loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> (Enc,Dec) trained Epoch: {}, SNR: {}, Average NLL loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))


def test_enc_dec_real_qam(model_enc, model_dec, model_mode, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, test_snr, input_data, target_data):
	model_enc.eval()
	model_dec.eval()
	test_ber=.0

	snr_test = test_snr

	num_test_batch = 100
	cur_idx = 0
	max_blocks = input_data.size()[0]
	with torch.no_grad():
		for batch_idx in range(num_test_batch):
			# FIX ME
			cur_idx = cur_idx % max_blocks
			
			data = input_data[cur_idx:cur_idx+batch_size,:]
			labels = target_data[cur_idx:cur_idx+batch_size,:]

			sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
			noise = sigma * torch.randn(data.shape).to(device)

			y = model_enc(data, noise)  
			
			estimate = model_dec(y)
			# breakpoint()
			estimate = torch.argmax(estimate,dim=2)

			test_ber += np.mean((estimate.cpu() != labels.cpu()).numpy())
			
			cur_idx += batch_size

		test_ber  /= 1.0*num_test_batch
		print("SNR : ", snr_test, ", BER:", float(test_ber), " ran a total of ", int(batch_size*max_blocks), " blocks ")
	model_enc.train()
	model_dec.train()
	return float(test_ber)
























































































def powerBlastBer_custom(batch_size, code_len, spa, SNRdB_vec):
	# batch_size = int(1e6)
	# code_len = 20
	p = spa

	pam_sym = [1, 0] # Skipping -1 for now; only 2 PAM
	prob = [p, 1-p]

	# adjust the constellation to have mean zero
	offset = p
	pam_sym = [x-offset for x in pam_sym]
	
	# Pass through channel
	# SNRdB_vec = np.arange(8.0,12.5,0.5)
	BERs = np.zeros_like(SNRdB_vec)

	num_batches = 1

	for i_s in range(SNRdB_vec.size):
		SNRdB = SNRdB_vec[i_s]
		sigma = torch.sqrt(torch.tensor(10**(-SNRdB/10)))
		num_err = 0

		for i_ib in range(num_batches):
			
			data = np.random.choice(pam_sym,(batch_size,code_len),p=prob)
			# FIX ME
			temp = int(batch_size/code_len)
			data = np.tile(np.eye(code_len),(temp,1)) - (p)           
			# breakpoint()
			data = torch.from_numpy(data).float()
			
			# scale such that expected power of symbol is 1
			pow_norm = torch.linalg.matrix_norm(data)/np.sqrt(code_len*batch_size) 
			x = data/pow_norm

			noise = sigma * torch.randn(x.shape[0],x.shape[1])

			y = x + noise

			# decode using MAP threshold rule
			c = pow_norm/p
			gamma = 0.5*(c - 1/c) + ((sigma**2)/(c + 1/c))*np.log((1-p)/p)
			
			# P = 1/pow_norm
			# gamma = P/2 + (1/P)*(sigma**2)*np.log(2*(1-p)/p)

			data_hat = torch.zeros_like(y)
			data_hat[y > gamma] = pam_sym[0]
			data_hat[y < gamma] = pam_sym[1]

			# check for total no. errors
			num_err = num_err + torch.sum(data_hat != data)
			
		err_rate = num_err/(code_len*batch_size*num_batches)
		BERs[i_s] = err_rate

	return BERs

def train_enc_dec_sk(epoch, model_enc, model_dec, input_data, model_type, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, train_snr):
	if (model_type == "enc"):
		model_enc.train()
	else:
		model_dec.train()
	train_loss = 0
	snr_train = train_snr
	num_train_batch = 100
	
	for batch_idx in tqdm(range(num_train_batch)):
		
		# data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)
		# data = torch.tensor(data).to(device).float()
		data = input_data[:,:,batch_idx].T
		# breakpoint()
		i_s = random.randint(0,len(snr_train)-1)
		sigma = torch.sqrt(torch.tensor(10**(-snr_train[i_s]/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		if (model_type == "enc"):
			optimizer_enc.zero_grad()
		else:
			optimizer_dec.zero_grad()
		
		y = model_enc(data,noise)
		
		estimate = model_dec(y)

		loss = F.binary_cross_entropy_with_logits(estimate, data)
		#loss = l(estimate, data)
		loss.backward()
		train_loss += loss.item()
		if (model_type == "enc"):
			optimizer_enc.step()
		else:
			optimizer_dec.step()
	if (model_type == "enc"):
		print('====> Enc trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> Dec trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))

def train_enc_dec(epoch, model_enc, model_dec, model_type, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, scheduler_enc, scheduler_dec, prob_train, train_snr):
	# if (model_type == "enc"):
	#     model_enc.train()
	# else:
	#     model_dec.train()

	model_enc.train()
	model_dec.train()
	
	train_loss = 0
	snr_train = train_snr
	num_train_batch = 1000
	
	# load training data
	torch.load('sk_data/sk_data_train.pt')
	breakpoint()
	
	# breakpoint()
	for batch_idx in tqdm(range(num_train_batch)):
		
		data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)
		# breakpoint()
		# temp = int(batch_size/block_len)
		# data = np.tile(np.eye(block_len),(temp,1))
		data = torch.tensor(data).to(device).float()
		i_s = random.randint(0,len(snr_train)-1)
		sigma = torch.sqrt(torch.tensor(10**(-snr_train[i_s]/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		# if (model_type == "enc"):
		#     optimizer_enc.zero_grad()
		# else:
		#     optimizer_dec.zero_grad()
		optimizer_enc.zero_grad()
		optimizer_dec.zero_grad()
		
		y = model_enc(data,noise)
		# breakpoint()
		estimate = model_dec(y)

		loss = F.binary_cross_entropy_with_logits(estimate, data)
		#loss = l(estimate, data)
		loss.backward()
		train_loss += loss.item()
		# if (model_type == "enc"):
		#     optimizer_enc.step()
		# else:
		#     optimizer_dec.step()
		optimizer_enc.step()
		optimizer_dec.step()
		# scheduler_enc.step(epoch + batch_idx/num_train_batch)
		# scheduler_dec.step(epoch + batch_idx/num_train_batch)
	if (model_type == "enc"):
		print('====> Enc trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> Dec trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))


def train_enc_dec_alt(epoch, model_enc, model_dec, model_type, alt_train, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, scheduler_enc, scheduler_dec, prob_train, train_snr):
	if (alt_train):
		if (model_type == "enc"):
			model_enc.train()
		else:
			model_dec.train()
	else:
		model_enc.train()
		model_dec.train()
	
	# load training data
	sk_data = torch.load('sk_data/sk_data_train.pt')
	breakpoint()
	
	train_loss = 0
	snr_train = train_snr
	num_train_batch = 1000
	# breakpoint()
	for batch_idx in tqdm(range(num_train_batch)):
		
		data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)
		# breakpoint()
		# temp = int(batch_size/block_len)
		# data = np.tile(np.eye(block_len),(temp,1))
		data = torch.tensor(data).to(device).float()
		i_s = random.randint(0,len(snr_train)-1)
		sigma = torch.sqrt(torch.tensor(10**(-snr_train[i_s]/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		if (alt_train):
			if (model_type == "enc"):
				optimizer_enc.zero_grad()
			else:
				optimizer_dec.zero_grad()
		else:
			optimizer_enc.zero_grad()
			optimizer_dec.zero_grad()
		
		y = model_enc(data,noise)
		# breakpoint()
		estimate = model_dec(y)

		loss = F.binary_cross_entropy_with_logits(estimate, data)
		#loss = l(estimate, data)
		loss.backward()
		train_loss += loss.item()
		if (alt_train):
			if (model_type == "enc"):
				optimizer_enc.step()
			else:
				optimizer_dec.step()
		else:
			optimizer_enc.step()
			optimizer_dec.step()
		# scheduler_enc.step(epoch + batch_idx/num_train_batch)
		# scheduler_dec.step(epoch + batch_idx/num_train_batch)
	if (alt_train):
		if (model_type == "enc"):
			print('====> Enc trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
		else:
			print('====> Dec trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> (Enc,Dec) trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))



































































def train_enc_dec_custom(epoch, model_enc, model_dec, model_type, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, scheduler_enc, scheduler_dec, prob_train, train_snr):
	# if (model_type == "enc"):
	#     model_enc.train()
	# else:
	#     model_dec.train()

	model_enc.train()
	model_dec.train()
	
	train_loss = 0
	snr_train = train_snr
	num_train_batch = 1000
	# breakpoint()
	for batch_idx in tqdm(range(num_train_batch)):
		
		data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)
		# breakpoint()
		temp = int(batch_size/block_len)
		data = np.tile(np.eye(block_len),(temp,1))
		data = torch.tensor(data).to(device).float()
		# breakpoint()
		i_s = random.randint(0,len(snr_train)-1)
		sigma = torch.sqrt(torch.tensor(10**(-snr_train[i_s]/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		# if (model_type == "enc"):
		#     optimizer_enc.zero_grad()
		# else:
		#     optimizer_dec.zero_grad()
		optimizer_enc.zero_grad()
		optimizer_dec.zero_grad()
		
		y = model_enc(data,noise)
		# breakpoint()
		estimate = model_dec(y)

		loss = F.binary_cross_entropy_with_logits(estimate, data)
		#loss = l(estimate, data)
		loss.backward()
		train_loss += loss.item()
		# if (model_type == "enc"):
		#     optimizer_enc.step()
		# else:
		#     optimizer_dec.step()
		optimizer_enc.step()
		optimizer_dec.step()
		# scheduler_enc.step(epoch + batch_idx/num_train_batch)
		# scheduler_dec.step(epoch + batch_idx/num_train_batch)
	if (model_type == "enc"):
		print('====> Enc trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> Dec trained Epoch: {}, SNR: {}, Average BCE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))


def test_enc_dec_custom(model_enc, model_dec, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, prob_test, test_snr):
	model_enc.eval()
	model_dec.eval()
	test_ber=.0

	snr_test = test_snr

	num_test_batch = 1000
	with torch.no_grad():
		for batch_idx in range(num_test_batch):

			data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_test)
			# FIX ME
			temp = int(batch_size/block_len)
			data = np.tile(np.eye(block_len),(temp,1))
			data = torch.tensor(data).to(device).float()

			sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
			noise = sigma * torch.randn(data.shape).to(device)

			optimizer_enc.zero_grad()
			optimizer_dec.zero_grad()
			
			y = model_enc(data, noise)
			estimate = model_dec(y)
			estimate = torch.sigmoid(estimate)
			estimate = torch.round(estimate)
			# breakpoint()
			test_ber += np.mean((estimate.cpu() != data.cpu()).numpy())
			# breakpoint()
			# test_ber  += errors_ber(estimate.cpu(),data.cpu())


		test_ber  /= 1.0*num_test_batch
		print("SNR : ", snr_test, ", BER:", float(test_ber))
	model_enc.train()
	model_dec.train()
	return float(test_ber)


		
def train_enc_dec_3pam(epoch, model_enc, model_dec, model_type, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, prob_train, train_snr):
	# if (model_type == "enc"):
	#     model_enc.train()
	# else:
	#     model_dec.train()
	model_enc.train()
	model_dec.train()
	train_loss = 0
	snr_train = train_snr
	num_train_batch = 1000
	
	for batch_idx in tqdm(range(num_train_batch)):
		
		data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_train)
		data = torch.tensor(data).to(device).float()
		i_s = random.randint(0,len(snr_train)-1)
		sigma = torch.sqrt(torch.tensor(10**(-snr_train[i_s]/10)))
		noise = sigma * torch.randn(data.shape).to(device)
		
		# if (model_type == "enc"):
		#     optimizer_enc.zero_grad()
		# else:
		#     optimizer_dec.zero_grad()
			
		optimizer_enc.zero_grad()
		optimizer_dec.zero_grad()
		
		y = model_enc(data,noise)
		
		estimate = model_dec(y)
		c_loss = nn.CrossEntropyLoss()
		loss = c_loss(estimate, data)
		breakpoint()
		#loss = l(estimate, data)
		loss.backward()
		train_loss += loss.item()
		if (model_type == "enc"):
			optimizer_enc.step()
		else:
			optimizer_dec.step()
	if (model_type == "enc"):
		print('====> Enc trained Epoch: {}, SNR: {}, Average CE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))
	else:
		print('====> Dec trained Epoch: {}, SNR: {}, Average CE loss: {:.8f}'.format(epoch, train_snr, train_loss /num_train_batch))        


def test_enc_dec(model_enc, model_dec, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, prob_test, test_snr):
	model_enc.eval()
	model_dec.eval()
	test_ber=.0

	snr_test = test_snr

	num_test_batch = 1000
	with torch.no_grad():
		for batch_idx in range(num_test_batch):

			data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_test)
			# temp = int(batch_size/block_len)
			# data = np.tile(np.eye(block_len),(temp,1))
			data = torch.tensor(data).to(device).float()

			sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
			noise = sigma * torch.randn(data.shape).to(device)

			optimizer_enc.zero_grad()
			optimizer_dec.zero_grad()
			
			y = model_enc(data, noise)
			estimate = model_dec(y)
			estimate = torch.sigmoid(estimate)
			estimate = torch.round(estimate)
			# breakpoint()
			test_ber += np.mean((estimate.cpu() != data.cpu()).numpy())
			# breakpoint()
			# test_ber  += errors_ber(estimate.cpu(),data.cpu())


		test_ber  /= 1.0*num_test_batch
		print("SNR : ", snr_test, ", BER:", float(test_ber))
	model_enc.train()
	model_dec.train()
	return float(test_ber)

def test_enc_dec_sk(model_enc, model_dec, input_data, pam_sym, batch_size, block_len, device, optimizer_enc, optimizer_dec, test_snr):
	model_enc.eval()
	model_dec.eval()
	test_ber=.0
	test_bler=.0
	
	snr_test = test_snr

	num_test_batch = int(input_data.shape[-1])

	with torch.no_grad():
		for batch_idx in range(num_test_batch):

			# data = np.random.choice(pam_sym,(batch_size,block_len),p=prob_test)
			# data = torch.tensor(data).to(device).float()
			data = input_data[:,:,batch_idx].T
			sigma = torch.sqrt(torch.tensor(10**(-snr_test/10)))
			noise = sigma * torch.randn(data.shape).to(device)

			optimizer_enc.zero_grad()
			optimizer_dec.zero_grad()
			
			y = model_enc(data, noise)
			estimate = model_dec(y)

			estimate = torch.sigmoid(estimate)
			estimate = torch.round(estimate)
			# breakpoint()
			test_ber += np.mean((estimate.cpu() != data.cpu()).numpy())
			# breakpoint()
			test_bler += np.mean(np.sum((estimate.cpu() != data.cpu()).numpy(),1) > 0)
			# breakpoint()
			# test_ber  += errors_ber(estimate.cpu(),data.cpu())


		test_ber  /= 1.0*num_test_batch
		test_bler  /= 1.0*num_test_batch
		print("SNR : ", snr_test, ", BER:", float(test_ber), ", BLER:", float(test_bler))
	return float(test_ber), float(test_bler)



# def errors_ber(y_true, y_pred):

#     t1 = np.round(y_true[:,:])
#     t2 = np.round(y_pred[:,:])

#     myOtherTensor = np.not_equal(t1, t2).float()
#     k = sum(sum(myOtherTensor))/(myOtherTensor.shape[0]*myOtherTensor.shape[1])
#     return k

# def errors_bler(y_true, y_pred):

#     y_true = y_true.view(y_true.shape[0], -1, 1)
#     y_pred = y_pred.view(y_pred.shape[0], -1, 1)

#     decoded_bits = torch.round(y_pred)
#     X_test       = torch.round(y_true)
#     tp0 = (abs(decoded_bits-X_test)).view([X_test.shape[0],X_test.shape[1]])
#     tp0 = tp0.cpu().numpy()

#     bler_err_rate = sum(np.sum(tp0,axis=1)>0)*1.0/(X_test.shape[0])

#     return bler_err_rate

	
	
	
	
	
	
	
















def _no_grad_trunc_normal_(tensor, mean, std, a, b):
	# Cut & paste from PyTorch official master until it's in a few official releases - RW
	# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
	def norm_cdf(x):
		# Computes standard normal cumulative distribution function
		return (1. + math.erf(x / math.sqrt(2.))) / 2.

	if (mean < a - 2 * std) or (mean > b + 2 * std):
		warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
					  "The distribution of values may be incorrect.",
					  stacklevel=2)

	with torch.no_grad():
		# Values are generated by using a truncated uniform distribution and
		# then using the inverse CDF for the normal distribution.
		# Get upper and lower cdf values
		l = norm_cdf((a - mean) / std)
		u = norm_cdf((b - mean) / std)

		# Uniformly fill tensor with values from [l, u], then translate to
		# [2l-1, 2u-1].
		tensor.uniform_(2 * l - 1, 2 * u - 1)

		# Use inverse cdf transform for normal distribution to get truncated
		# standard normal
		tensor.erfinv_()

		# Transform to proper mean, std
		tensor.mul_(std * math.sqrt(2.))
		tensor.add_(mean)

		# Clamp to ensure it's in the proper range
		tensor.clamp_(min=a, max=b)
		return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
	return _no_grad_trunc_normal_(tensor, mean, std, a, b)