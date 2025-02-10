import os
import pdb
import numpy as np
import scipy.special

from tqdm import tqdm
import torch
import time

def qfunction(z):
	return(0.5-0.5*scipy.special.erf(z/np.sqrt(2)))

def Powercheck(x):
	return (np.mean(np.abs(x)**2))

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

def list_demod(X,PAMSize,l=3):
	# offset correction factor
	offset_factor = 2**PAMSize - 1

	int_min = int(np.min(X)) - 2
	int_max = int(np.max(X)) + 2
	grid = np.arange(int_min, int_max)
 
	diff = np.abs(X[..., np.newaxis] - grid)
	X_ind = np.argsort(diff, axis=2)[:,:, :l]
	X_int = grid[X_ind]
	# limit the values
	X_int = np.minimum(offset_factor,np.maximum(0,X_int))
	# rehape to (l, batch_size, block_len)
	X_list = np.transpose(X_int, (2, 0, 1))

	return X_list

def PAMdemodulate(Y,PAMSize,l):
	# offset correction and scaling
	offset = 2**PAMSize - 1
	eta = np.sqrt(3 / ((2 ** (2*PAMSize)) - 1) )
 
	Xh = np.round((Y/eta + offset)/2).astype(int)
	Xh = np.minimum(offset,np.maximum(0,Xh))

	Xl = list_demod((Y/eta + offset)/2,l)
	
	Xb = np.zeros((Y.shape[0],Y.shape[1]*PAMSize))
	
	Xb = np.unpackbits(Xh.astype(np.uint8).reshape(-1, 1), axis=1)[:, -PAMSize:]
	Xb = Xb.reshape(Xh.shape[0], -1)

	return Xb, Xh, Xl

def SK_scheme(Theta,PAMSize,num_rounds,SNR):
	sigma2chan = 1/SNR
	sigma2n = 1/(1+SNR) # LMMSE estimate error
	alphan = 1/np.sqrt(sigma2n)
	
	# X = Theta
	X = Theta
	# print(Powercheck(X))
	Y = AWGNChan(X,SNR)

	K = Theta.shape[1]*PAMSize
	ThetaHat = Y/(1+sigma2chan)
	
	l = 3	
	Theta_hat = np.zeros((Theta.shape[0],Theta.shape[1],num_rounds))
	Theta_ind_hat = np.zeros((Theta.shape[0],Theta.shape[1],num_rounds))
	Theta_ind_hat_list = np.zeros((l,Theta.shape[0],Theta.shape[1],num_rounds))
	bits_hat = np.zeros((Theta.shape[0],K,num_rounds))

	Theta_hat[:,:,0] = ThetaHat
	
	bits_hat[:,:,0], Theta_ind_hat[:,:,0], Theta_ind_hat_list[:,:,:,0] = PAMdemodulate(ThetaHat,PAMSize,l)
	pow_total = Powercheck(X)
	sim_start = time.process_time()
	dec_time = 0
	enc_time = 0
	for i_r in range(1,num_rounds):
		
		enc_start = time.process_time()
  
		errn = ThetaHat-Theta

		X = alphan*errn
		
		# print(Powercheck(X))
		# pow_total = pow_total + Powercheck(X)
		Y = AWGNChan(X,SNR)
		
		# advance
		betan = np.sqrt(sigma2n/sigma2chan) * np.sqrt(SNR)/(1+SNR)

		ThetaHat = ThetaHat - Y*betan
		sigma2n = sigma2n/(1+SNR)
		alphan = 1/np.sqrt(sigma2n)
  
		enc_end = time.process_time()
  
		enc_time = enc_time + enc_end - enc_start
		
		# save data of all rounds
		Theta_hat[:,:,i_r] = ThetaHat
		
		dec_start = time.process_time()
		bits_hat[:,:,i_r], Theta_ind_hat[:,:,i_r], Theta_ind_hat_list[:,:,:,i_r] = PAMdemodulate(ThetaHat,PAMSize,l)
		dec_end = time.process_time()
		dec_time = dec_time + dec_end - dec_start
	sk_end = time.process_time()
 
	print("Time taken for SK: ", sk_end - sim_start)
	throughput = Theta.shape[0]/(sk_end - sim_start)
	print("Throughput: ", throughput)
 
	dec_throughput = Theta.shape[0]/dec_time
	print("Decoding Throughput: ", dec_throughput)
 
	enc_throughput = Theta.shape[0]/enc_time
	print("Encoding Throughput: ", enc_throughput)
 
	breakpoint()
	# print(pow_total/num_rounds)
	# breakpoint()
	return bits_hat[:,:,-1], Theta_hat, Theta_ind_hat, Theta_ind_hat_list, bits_hat, Theta_ind_hat[:,:,-1]


def generate_sk_data(K,PAMSize,num_rounds,num_batches,batch_size,SNRdB):
	
	PAMSize = int(np.log2(PAMSize))
 	
	block_len = int(K/PAMSize)

	BER = 0

	Theta_data = np.zeros((block_len,batch_size,num_batches))
	Theta_ind_data = np.zeros((block_len,batch_size,num_batches))
	bits_data = np.zeros((K,batch_size,num_batches))
	
	l = 3
	Theta_hat_data = np.zeros((block_len,batch_size,num_batches,num_rounds))
	Theta_ind_hat_data = np.zeros((block_len,batch_size,num_batches,num_rounds))
	# Theta_ind_hat_list_data = np.zeros((l,block_len,batch_size,num_batches,num_rounds))
	# bits_hat_data = np.zeros((K,batch_size,num_batches,num_rounds))
	ser_rounds = np.zeros((num_rounds,1))
	bler_rounds = np.zeros((num_rounds,1))
 
	p = 0.5
	bits_ref = [0, 1]
	prob = [p, 1-p]

	SNR = 10**(SNRdB/10)
	
	berr_snr = 0
	serr_snr = 0

	for i_n in tqdm(range(num_batches)):
	   
		inbits = np.random.choice(bits_ref,(batch_size,K),p=prob)
		# breakpoint()
		Theta, X_ref = PAMmodulate(inbits,PAMSize)
		
		# store data
		bits_data[:,:,i_n] = inbits.T
		Theta_data[:,:,i_n] = Theta.T
		Theta_ind_data[:,:,i_n] = X_ref.T

		outbits, Theta_hat, Theta_ind_hat, Theta_ind_hat_list, bits_hat, Xh = SK_scheme(Theta,PAMSize,num_rounds,SNR)

		Theta_hat_data[:,:,i_n,:]= Theta_hat.transpose((1,0,2))
		Theta_ind_hat_data[:,:,i_n,:]= Theta_ind_hat.transpose((1,0,2))
		# Theta_ind_hat_list_data[:,:,:,i_n,:]= Theta_ind_hat_list.transpose((0,2,1,3))
		# bits_hat_data[:,:,i_n,:] = bits_hat.transpose((1,0,2))

		berr_snr = berr_snr + np.sum(outbits != inbits)
		serr_snr = serr_snr + np.sum(Xh != X_ref)

	for jj in range(num_rounds):
		ser_rounds[jj] = np.mean((Theta_ind_hat_data[:,:,:,jj] != Theta_ind_data).astype(float))
		bler_rounds[jj] = np.mean(np.sum((Theta_ind_hat_data[:,:,:,jj] != Theta_ind_data).astype(float),0) > 0)

	BER = berr_snr/(K*num_batches*batch_size)
	SER = serr_snr/(block_len*num_batches*batch_size)
	
	return Theta_data,  Theta_ind_data, Theta_hat_data, Theta_ind_hat_data, BER, SER, ser_rounds, bler_rounds
	# return bits_data, bits_hat_data, Theta_data,  Theta_ind_data, Theta_hat_data, Theta_ind_hat_data, Theta_ind_hat_list_data, BER, SER, ser_rounds, bler_rounds


