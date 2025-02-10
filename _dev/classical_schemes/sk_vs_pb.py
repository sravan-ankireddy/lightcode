import os
import pdb
import numpy as np

np.random.seed(0)
import time
from tqdm import tqdm
from generate_sk import AWGNChan, Powercheck, generate_sk_data
from utils import powerblastErf
import matplotlib.pyplot as plt
import pickle
import torch

import logging

num_rounds = 10 # num_rounds of SK vs (num_rounds - num_rounds_pb) rounds of SK + num_rounds_pb rounds of power blast
num_rounds_pb = 1

if (num_rounds == 9):
    snr_start = -1.5
    snr_end = -0.5
elif(num_rounds == 8):
    snr_start = 0.0
    snr_end = 0.0
elif(num_rounds == 7):
    snr_start = 1.0
    snr_end = 1.0
elif(num_rounds == 6):
    snr_start = 1.0
    snr_end = 3.0
elif(num_rounds == 5):
    snr_start = 2.0
    snr_end = 3.0 

snr_start = -1.0
snr_end = -1.0

num_snr = 1

SNRdB_vec = np.linspace(snr_start, snr_end, num_snr)
SNRdB_vec = np.round(SNRdB_vec,2)

BER_vec_sk = np.zeros_like(SNRdB_vec)
SER_vec_sk = np.zeros_like(SNRdB_vec)
BLER_vec_sk = np.zeros_like(SNRdB_vec)

BER_vec_pb = np.zeros((num_rounds_pb,len(SNRdB_vec)))
SER_vec_pb = np.zeros((num_rounds_pb,len(SNRdB_vec)))
BLER_vec_pb = np.zeros((num_rounds_pb,len(SNRdB_vec)))

BER_vec_pb_th = np.zeros((num_rounds_pb,len(SNRdB_vec)))
SER_vec_pb_th = np.zeros((num_rounds_pb,len(SNRdB_vec)))
BLER_vec_pb_th = np.zeros((num_rounds_pb,len(SNRdB_vec)))

SER_vec_sk_rounds = np.zeros((len(SNRdB_vec),num_rounds))
BLER_vec_sk_rounds = np.zeros((len(SNRdB_vec),num_rounds))

spa_0_vec = np.zeros_like(SNRdB_vec)
spa_1_vec = np.zeros_like(SNRdB_vec)
spa_2_vec = np.zeros_like(SNRdB_vec)

num_batches = 10
batch_size = int(1e6)
num_samples = int(num_batches*batch_size)

K = 3
PAMSize = 3
sym_len = int(K/PAMSize)

# if (PAMSize == 4):
#     SNRdB_vec = SNRdB_vec + 1.7
#     SNRdB_vec = np.round(SNRdB_vec,2)
    
list_decoding = 0

results_folder = "results_sk_ref"

log_folder = f'{results_folder}/logs_sk_{num_rounds}'

# # append date and time to log folder
# date_time = time.strftime("%Y_%m_%d-%H_%M_%S")

# log_folder = f'{log_folder}/{date_time}'

if not os.path.exists(results_folder):
    os.makedirs(results_folder)

if not os.path.exists(log_folder):
    os.makedirs(log_folder)

save_data = 0


# folder for saving data
# create folders if not existing
sk_data_path = f'sk_data/{num_samples}'
if not (os.path.exists(sk_data_path)) and save_data:
   os.makedirs(sk_data_path)


# create log file
log_file_name = f'{log_folder}/log_sk_vs_pb_sk_{num_rounds}_pb_{num_rounds_pb}_ls_{list_decoding}_ref_{num_samples}_{PAMSize}_PAM_SNR_{SNRdB_vec[0]}_{SNRdB_vec[-1]}.txt'

logging.basicConfig(format='%(message)s', filename=log_file_name, encoding='utf-8', level=logging.INFO)

    
start = time.process_time()

print(f'Simulating {num_samples} blocks per SNR : {K} bits, {sym_len} symbols\n')
logging.info(f'Simulating {num_samples} blocks per SNR : {K} bits, {sym_len} symbols\n')

full_start = time.process_time()
sk_data_train = []
for i_s in tqdm(range(SNRdB_vec.size)):

    SNRdB = SNRdB_vec[i_s]
    SNR = 10**(SNRdB/10)
    
    # SNR= SNR*((1+SNR)**5)
    
    # SNRdB = 10*np.log10(SNR)
    
    # FIX ME : usinf effective sigma
    # SNR = SNR*((1+SNR)**8)
    start = time.process_time()
    # bits_data, bits_hat_data, Theta_data, Theta_ind_data, Theta_hat_data, Theta_ind_hat_data, Theta_ind_hat_list_data, train_ber, train_ser, ser_rounds, bler_rounds = generate_sk_data(K,2**PAMSize,num_rounds,num_batches,batch_size,SNRdB)
    
    Theta_data, Theta_ind_data, Theta_hat_data, Theta_ind_hat_data, train_ber, train_ser, ser_rounds, bler_rounds = generate_sk_data(K,2**PAMSize,num_rounds,num_batches,batch_size,SNRdB)
    
    
    
    end = time.process_time()
    print(f'Elapsed time: {end-start} seconds')
    logging.info(f'Elapsed time: {end-start} seconds')
    
    # save relevant data
    sk_data = []
    
    # list_size = Theta_ind_hat_list_p.shape[0]
    list_size = 2
    pos_vec = np.full((Theta_ind_data.shape[0],Theta_ind_data.shape[1]), list_size)

    BER_vec_sk[i_s] = train_ber
    SER_vec_sk[i_s] = train_ser
    SER_vec_sk_rounds[i_s,:] = ser_rounds.T
    BLER_vec_sk_rounds[i_s,:] = bler_rounds.T
    
    # Theoretical power blast in final for error in indices, assumes list decoding, 1 step error: calculate excpected error rate using erf
    sigmasqn = np.power(10,-SNRdB/10).astype(np.float64)
    sigman = np.sqrt(sigmasqn)
    
    for n_pb in range(1,num_rounds_pb+1):
        # start of power blast for n_pb rounds
        ThetaInd = Theta_ind_data
        ThetaHat = Theta_hat_data[:,:,:,-(n_pb+1)]
        ThetaIndHat = Theta_ind_hat_data[:,:,:,-(n_pb+1)]
        
        # compute theoreical SER of powerblast assuming 1 step error for 1 to num_rounds_pb rounds of power blast
        if (list_decoding == 1):
            err_vec = np.sign(pos_vec)
            spa = np.mean(err_vec)
        else:
            err_vec = np.sign(ThetaInd - ThetaIndHat)
            spa = np.mean(np.abs(err_vec))
        
        for i_pb in range(n_pb):
            if (spa > 0):
                cur_ser = powerblastErf(1.0,spa,sigman,list_decoding)
                cur_bler = 1 - (1 - cur_ser)**sym_len
            else:
                cur_ser = 0
                cur_bler = 0
            spa = cur_ser
        SER_vec_pb_th[n_pb-1, i_s] = cur_ser
        BLER_vec_pb_th[n_pb-1, i_s] = cur_bler

    # compute and store the reference error vector: default is with respect to the round before powerblast
    if list_decoding:
        err_ref = pos_vec 
    else:
        ThetaIndHat = Theta_ind_hat_data[:,:,:,num_rounds-num_rounds_pb-1]
        err_ref = ThetaInd - ThetaIndHat

    # compute the stats of error with magnitude > 1 in pos_vec
    if list_decoding == 1:
        err_ref_large_1 = np.abs(err_ref) > 1
        err_ref_large_2 = np.abs(err_ref) > 2
        print(f'SNRdB {SNRdB}, List decoding after {num_rounds-num_rounds_pb} rounds of SK: Prob(error mag. > 1): {np.mean(err_ref_large_1)}, Prob(error mag. > 2): {np.mean(err_ref_large_2)}')
        logging.info(f'SNRdB {SNRdB}, List decoding after {num_rounds-num_rounds_pb} rounds of SK: Prob(error mag. > 1): {np.mean(err_ref_large_1)}, Prob(error mag. > 2): {np.mean(err_ref_large_2)}')
    else:
        err_ref_large_1 = np.abs(err_ref) > 1
        err_ref_large_2 = np.abs(err_ref) > 2
        err_ref_large_3 = np.abs(err_ref) > 3
        print(f'SNRdB {SNRdB}, MAP decoding after {num_rounds-num_rounds_pb} rounds of SK: Prob(error mag. > 1): {np.mean(err_ref_large_1)}, Prob(error mag. > 2): {np.mean(err_ref_large_2)}, Prob(error mag. > 3): {np.mean(err_ref_large_3)}')
        logging.info(f'SNRdB {SNRdB}, MAP decoding after {num_rounds-num_rounds_pb} rounds of SK: Prob(error mag. > 1): {np.mean(err_ref_large_1)}, Prob(error mag. > 2): {np.mean(err_ref_large_2)}, Prob(error mag. > 3): {np.mean(err_ref_large_3)}')
        
    if save_data:
        torch.save(sk_data,f'{sk_data_path}/sk_data_train_sk_{num_rounds}_pb_{num_rounds_pb}_K_{K}_{sym_len}_snr_{SNRdB}.pt') 
        
    # FIX ME : more than 1 rounds of power blast has incorrect implementation   
    # FIX ME : supporting only 1 step error for now
    # Empirical simulation of power blast for last num_rounds_pb rounds instead of sk
    for n_pb in range(1,num_rounds_pb+1):
        if (n_pb > 0):       
            # extract data to pass to power blast for n_pb rounds
            ThetaInd = Theta_ind_data
            Theta = Theta_data[:,:,:]
            ThetaHat = Theta_hat_data[:,:,:,-(n_pb+1)]
            ThetaIndHat = Theta_ind_hat_data[:,:,:,-(n_pb+1)]
            
            # running err vector initialisation : imposing approximations
            if list_decoding:
                err_ref = pos_vec
                err = err_ref #np.sign(err_ref)  # error is among the first 2 values in the list
            else:
                err_ref = ThetaInd - ThetaIndHat
                err = np.sign(err_ref)  # error belongs to {-1,0,1}
                
            spa = np.mean(np.abs(err))
                
            err_est = np.zeros_like(err)
            for i_pb in range(n_pb):
                
                spa = np.mean(np.abs(err))
                if (spa > 0):
                    # shift x to have zero mean
                    x = err - np.mean(err)
                        
                    pow_norm = np.linalg.norm(x)/np.sqrt(sym_len*batch_size*num_batches) 
                    x = x/pow_norm

                    # print(Powercheck(x))
                    y = AWGNChan(x,SNR)

                    # decode using MAP threshold rule 
                    sigma = np.sqrt(1/SNR)
                    c = pow_norm/spa
                    P = np.sqrt(1/spa)
                    if list_decoding:                            
                        # decode using MAP threshold rule : list decoding available
                        c = pow_norm/spa
                        A = c - 1/c
                        B = c + 1/c

                        sigmasqn = sigma**2
                        gamma = (A/2) + (sigmasqn/B)*np.log((1-spa)/spa)
                        
                        # gamma = P/2 + (1/P)*(sigma**2)*np.log((1-spa)/spa)
                        err_hat = np.zeros_like(y)

                        ## FIX ME: 
                        gamma_2 = (2-np.mean(err))/pow_norm - gamma
                        err_hat[y > gamma] = 1
                        err_hat[y > gamma_2] = 2

                        # # update error estimate compute new error
                        # err_est = np.mod(err_est + err_hat,2)
                        # err = np.abs(np.sign(err_ref - err_est))
                        # update error estimate compute new error
                        err_est = err_est + err_hat
                        err = err_ref - err_est
                        
                        train_ser = np.mean(err != 0)
                        train_bler = np.mean(np.sum(err != 0,0) > 0)
                    else:
                        c = np.sqrt(1/spa)
                        gamma = c/2 + (sigmasqn/c)*np.log(2*(1-spa)/spa)
                        
                        err_hat = np.zeros_like(y)
                                            
                        err_hat[y > gamma] = 1
                        err_hat[y < -gamma] = -1

                        # update error estimate and compute new error
                        err_est = err_est + err_hat
                        err = np.sign(err_ref - err_est)
                        
                        train_ser = np.mean(err != 0)
                        train_bler = np.mean(np.sum(err != 0,0) > 0)
                else:
                    train_ser =  0
                    train_bler = 0
                    break
            # breakpoint()
            # final SER
            SER_vec_pb[n_pb-1, i_s] = train_ser
            BLER_vec_pb[n_pb-1, i_s] = train_bler
            
            print(f'SNRdB {SNRdB}, Empirical SER after {num_rounds-num_rounds_pb} rounds of SK + {n_pb} rounds of power blast: {train_ser}')
            print(f'SNRdB {SNRdB}, Empirical BLER after {num_rounds-num_rounds_pb} rounds of SK + {n_pb} rounds of power blast: {train_bler}')
            
            logging.info(f'SNRdB {SNRdB}, Empirical SER after {num_rounds-num_rounds_pb} rounds of SK + {n_pb} rounds of power blast: {train_ser}')
            logging.info(f'SNRdB {SNRdB}, Empirical BLER after {num_rounds-num_rounds_pb} rounds of SK + {n_pb} rounds of power blast: {train_bler}')

            
SER_vec_sk_ref = SER_vec_sk
BLER_vec_sk_ref = BLER_vec_sk_rounds[:,-1] 

print("Time taken : ")
print(time.process_time() - full_start)

# compute throughput
time_taken = time.process_time() - full_start
throughput = num_samples/time_taken

print(f'Throughput : {throughput} samples per second')

print(f"SNR vec :{SNRdB_vec}")

print("\nBER after %d rounds of SK: \n" % (num_rounds))
print(repr(BER_vec_sk.T))

print("\nAfter %d rounds of SK - SER : \n" % (num_rounds))
print(repr(SER_vec_sk.T))

print("\nAfter %d rounds of SK - SER : \n" % (num_rounds-1))
print(repr(SER_vec_sk_rounds[:,-2]))

print("\nAfter %d rounds of SK - SER : \n" % (num_rounds-2))
print(repr(SER_vec_sk_rounds[:,-3]))

if (num_rounds_pb > 0):
    print(f"\n Ran {num_samples} blocks per SNR point")
    print("\nAfter %d rounds of SK + %d round of power blast - empirical SER : \n" % (num_rounds-num_rounds_pb,num_rounds_pb))
    print(repr(SER_vec_pb.T))

    print("\nAfter %d rounds of SK + %d round of power blast - theoretical SER : \n" % (num_rounds-num_rounds_pb,num_rounds_pb))
    print(repr(SER_vec_pb_th.T))
    
if list_decoding:
    print(f'Spa 0: {spa_0_vec}')
    print(f'Spa 1: {spa_1_vec}')
    print(f'Spa 1+: {spa_2_vec}')
    
    
# logging
logging.info("Time taken : ")
logging.info(time.process_time() - start)

logging.info("\nBER after %d rounds of SK: \n" % (num_rounds))
logging.info(repr(BER_vec_sk.T))

logging.info("\nAfter %d rounds of SK - SER : \n" % (num_rounds))
logging.info(repr(SER_vec_sk.T))

logging.info("\nAfter %d rounds of SK - SER : \n" % (num_rounds-1))
logging.info(repr(SER_vec_sk_rounds[:,-2]))

logging.info("\nAfter %d rounds of SK - SER : \n" % (num_rounds-2))
logging.info(repr(SER_vec_sk_rounds[:,-3]))

if (num_rounds_pb > 0):
    logging.info(f"\n Ran {num_samples} blocks per SNR point")
    logging.info("\nAfter %d rounds of SK + %d round of power blast - empirical SER : \n" % (num_rounds-num_rounds_pb,num_rounds_pb))
    logging.info(repr(SER_vec_pb.T))

    logging.info("\nAfter %d rounds of SK + %d round of power blast - theoretical SER : \n" % (num_rounds-num_rounds_pb,num_rounds_pb))
    logging.info(repr(SER_vec_pb_th.T))
    
if list_decoding:
    logging.info(f'Spa 0: {spa_0_vec}')
    logging.info(f'Spa 1: {spa_1_vec}')
    logging.info(f'Spa 1+: {spa_2_vec}')


    
# plots
plot_and_marker_styles = ['b-o','b-s','b-d','g-o','r-s','r-d','k-o','k-s','k-d']

plt.figure(1)
# plot sk results
num_sk_plots = 3
leg = []
for i_sk in range(num_sk_plots):
    plt.semilogy(SNRdB_vec, SER_vec_sk_rounds[:,-(i_sk+1)], plot_and_marker_styles[i_sk])
    leg.append(f'{num_rounds-i_sk} rounds of SK')

# plot empirical power blast results
plot_and_marker_styles =  ['g-o','g-s','g-d']    
for i_pb in range(num_rounds_pb):
    plt.semilogy(SNRdB_vec, SER_vec_pb[i_pb,:], plot_and_marker_styles[i_pb])
    leg.append(f'{i_pb+1} round of power blast -- empirical')
    
# plot theoretical power blast results
plot_and_marker_styles =  ['r-o','r-s','r-d']  
for i_pb in range(num_rounds_pb):
    plt.semilogy(SNRdB_vec, SER_vec_pb_th[i_pb,:], plot_and_marker_styles[i_pb])
    leg.append(f'{i_pb+1} round of power blast -- theoretical')

plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.grid(True, which="both")

title  = f'SER: {num_rounds-num_rounds_pb} rounds of SK + {num_rounds_pb} round of power blast'
if list_decoding:
    title  = f'{title} with list decoding'

plt.legend(leg)

plt.title(title)
plot_name = results_folder + f'/ser_sk_vs_pb_sk_{num_rounds}_pb_{num_rounds_pb}_ls_{list_decoding}_ref_{num_samples}_{PAMSize}_PAM_SNR_{SNRdB_vec[0]}_{SNRdB_vec[-1]}.png'
plt.savefig(plot_name)