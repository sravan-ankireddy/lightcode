import torch,os
from torch.autograd import Variable

from nn_layers import *
import math

class PositionalEncoder_fixed(nn.Module):
    def __init__(self, lenWord=32, max_seq_len=200, dropout=0.0):
        super().__init__()
        self.lenWord = lenWord
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, lenWord)
        for pos in range(max_seq_len):
            for i in range(0, lenWord, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / lenWord)))
                if lenWord != 1:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / lenWord)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.lenWord)
        seq_len = x.size(1)
        pe = Variable(self.pe[:, :seq_len], requires_grad=False)
        x = x + pe
        return self.dropout(x)
        
class DeepVLF(nn.Module):
    def __init__(self, args):
        super(DeepVLF, self).__init__()
        self.args = args
        self.truncated = args.truncated
        self.pe = PositionalEncoder_fixed()
        ######### Initialize encoder and decoder ###############
        self.Tmodel = Transformer(mod="trx",
                           input_size=args.block_size+2*(args.truncated-1), 
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           )
        
        self.Rmodel = Transformer(mod="rec",
                           input_size=args.block_class+args.truncated,
                           block_size=args.block_size, 
                           d_model=args.d_model_trx, 
                           N=args.N_trx+1, 
                           heads=args.heads_trx, 
                           dropout=args.dropout, 
                           custom_attn=args.custom_attn,
                           multclass=args.multclass,
                           )


    def power_constraint(self, inputs, isTraining, eachbatch, idx=0, direction='fw'):
        # direction = 'fw' or 'fb'
        if isTraining == 1:
            # training
            this_mean = torch.mean(inputs, 0)
            this_std = torch.std(inputs, 0)
        elif isTraining == 0:
            # test
            if eachbatch == 0:
                this_mean = torch.mean(inputs, 0)
                this_std = torch.std(inputs, 0)
                if not os.path.exists('statistics'):
                    os.mkdir('statistics')
                torch.save(this_mean, 'statistics/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/this_std' + str(idx) + direction)
            elif eachbatch <= 100:
                this_mean = torch.load('statistics/this_mean' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.mean(inputs, 0) / (eachbatch + 1)
                this_std = torch.load('statistics/this_std' + str(idx) + direction) * eachbatch / (
                            eachbatch + 1) + torch.std(inputs, 0) / (eachbatch + 1)
                torch.save(this_mean, 'statistics/this_mean' + str(idx) + direction)
                torch.save(this_std, 'statistics/this_std' + str(idx) + direction)
            else:
                this_mean = torch.load('statistics/this_mean' + str(idx) + direction)
                this_std = torch.load('statistics/this_std' + str(idx) + direction)

        outputs = (inputs - this_mean) * 1.0 / (this_std + 1e-8)
        return outputs

    ########### IMPORTANT ##################
    # We use unmodulated bits at encoder
    #######################################
    def forward_train(self, belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par, ys, optimizer):
        combined_noise_par = fwd_noise_par + fb_noise_par
        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, 
                           self.args.numb_block,dtype=torch.bool).to(self.args.device)
        train_log = []
        es=[]
        losses = torch.tensor(0.).to(self.args.device)
        ##############Define lower bound of communication rounds######################
        if belief_threshold<=0.99999:
            mu = 5
        elif 0.99999<belief_threshold<=0.999999:
            mu = 6
        else:
            mu = 7
        eta = 10**(self.args.snr1/10)
        tau_plus = max(mu,round(2*self.args.block_size /math.log((1 + eta),2)))

        for idx in range(self.truncated):
            optimizer.zero_grad()
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     combined_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe,idx,self.args.tau_vd)
            parity = self.power_constraint(output,
                                           eachbatch=eachbatch,
                                           idx=idx,
                                           isTraining=1)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe,idx,self.args.tau_vd)
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)

            if idx+1>=tau_plus:
                ############# Backwarding and update gradient ###################################################
                preds = torch.log(belief.contiguous().view(-1, belief.size(-1)))
                mask_flatten = mask.view(-1).to(self.args.device)
                loss = F.nll_loss(preds[~mask_flatten], ys.to(self.args.device)[~mask_flatten])
                loss_cof = 10**(idx+1-self.args.offset)
                losses += loss_cof*loss
                ############# Update the decoding decision ###################################################
                mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(self.args.batchSize,
                                                                                     self.args.numb_block,
                                                                                     dtype=torch.bool).to(self.args.device)
                if self.args.break_trained and mask.all():
                    break
                ############# logging early_stop ###################################################
                early_stop = torch.sum(mask) - sum(es[:idx])
                es.append(early_stop.item())
                train_log.append({"round":idx,"loss":loss.item(),"early_stop":early_stop.item()})
            else:
                train_log.append({"round": idx, "loss": None, "early_stop": 0})
        losses.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip_th)
        optimizer.step()
        return train_log,preds,losses.item()
    

    def forward_evaluate(self, belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par):
        combined_noise_par = fwd_noise_par + fb_noise_par
        bVec_md = 2*bVec-1
        belief = torch.full((self.args.batchSize, 
                             self.args.numb_block, 
                             self.args.block_class), 
                             fill_value=1 /self.args.block_class,
                             requires_grad=False).to(self.args.device)
        mask = torch.zeros(self.args.batchSize, self.args.numb_block,dtype=torch.bool).to(self.args.device)
        test_log = []
        es = []
        ##############Define lower bound of communication rounds######################
        if belief_threshold <= 0.99999:
            mu = 5
        elif 0.99999 < belief_threshold <= 0.999999:
            mu = 6
        else:
            mu = 7
        eta = 10 ** (self.args.snr1 / 10)
        tau_plus = max(mu,round(2*self.args.block_size/math.log((1 + eta),2)))


        for idx in range(self.truncated):
            ############# Generate the input features ###################################################
            if idx == 0: # phase 0
                src = torch.cat([bVec_md,torch.zeros(self.args.batchSize, 
                                                     self.args.numb_block,
                                                     2*(self.truncated-1)).to(self.args.device)],dim=2)
            else:
                src_new = torch.cat([bVec_md, 
                                     parity_all,
                                     torch.zeros(self.args.batchSize, self.args.numb_block, self.truncated-(idx+1)).to(self.args.device),
                                     combined_noise_par[:, :, :idx],
                                     torch.zeros(self.args.batchSize,self.args.numb_block, self.truncated - (idx + 1)).to(
                                         self.args.device)],dim=2)
                src = torch.where(mask.unsqueeze(2),src,src_new)

            ############# Generate the parity ###################################################
            output = self.Tmodel(src, None,self.pe,idx,self.args.tau_vd)
            parity = self.power_constraint(output, 
                                           eachbatch=eachbatch, 
                                           idx=idx, 
                                           isTraining=0)

            ############# Generate the received symbols ###################################################
            if idx == 0:
                parity_all = parity
                received = torch.cat([parity + fwd_noise_par[:,:,0].unsqueeze(-1),
                                      torch.zeros(self.args.batchSize, self.args.numb_block,self.truncated-1).
                                to(self.args.device),belief], dim= 2)
            else:
                parity_all = torch.cat([parity_all, parity], dim=2)
                received_new = torch.cat([parity_all+ fwd_noise_par[:,:,:idx+1],
                                          torch.zeros(self.args.batchSize,self.args.numb_block,self.truncated-(1+idx)).to(self.args.device),
                                          belief], dim = 2)
                received = torch.where(mask.unsqueeze(2),received,received_new)

            ############# Update the received beliefs ###################################################
            belief_new = self.Rmodel(received, None,self.pe,idx,self.args.tau_vd)
            belief = torch.where(mask.unsqueeze(2), belief, belief_new)

            if idx+1>=tau_plus:
                ############# Update the decoding decision ###################################################
                mask = (torch.max(belief, dim=2)[0] > belief_threshold) & torch.ones(self.args.batchSize,
                                                                                     self.args.numb_block,
                                                                                     dtype=torch.bool).to(self.args.device)
                if mask.all():
                    break
                ############# logging early_stop ###################################################
                early_stop = torch.sum(mask) - sum(es[:idx])
                es.append(early_stop.item())
                test_log.append({"round":idx,"early_stop":early_stop.item()})
            else: 
                test_log.append({"round": idx,  "early_stop": 0})
        return test_log,belief

    def forward(self,belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par,ys,isTraining=1):
        if isTraining:
            optimizer=self.args.optimizer
            return self.forward_train(belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par, ys,optimizer)
        else:
            return self.forward_evaluate(belief_threshold, eachbatch, bVec, fwd_noise_par,fb_noise_par)
        


