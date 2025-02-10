import torch, pdb, os
from nn_layers import *
from parameters import *
import numpy as np
import torch.optim as optim
from model import DeepVLF

def compute_avgcodelength(logs):
    es_list = []
    for log in logs:
        es_list.append(log['early_stop'])
    avg_codelen = 0
    for idx in range(len(logs)):
        avg_codelen+= es_list[idx]*(idx+1)
    avg_codelen = (avg_codelen + args.truncated* ((args.batchSize*args.numb_block) - sum(es_list)))/(args.batchSize*args.numb_block)
    return avg_codelen
def ModelAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def train_model(model, args):
    print(args)
    print("-->-->-->-->-->-->-->-->-->--> start training ...")
    model.train()
    map_vec = torch.tensor([1,2,4])# maping block of bits to class label
    ###################Setting optimizer############################
    if args.opt_method == 'adamW':
            args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd, amsgrad=False)
    elif args.opt_method == 'lamb':
        args.optimizer = optim.Lamb(model.parameters(),lr= 1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.wd)
    else:
        args.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    if args.use_lr_schedule:
        lambda1 = lambda epoch: (1-epoch*args.core/(args.total_iter-args.start_step))
        args.scheduler = torch.optim.lr_scheduler.LambdaLR(args.optimizer, lr_lambda=lambda1)
    ######################## resume training ####################################################
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        if args.train and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            args.optimizer.load_state_dict(checkpoint['optimizer'])
            args.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_step = checkpoint['epoch'] + 1
        print("================================ Successfully load the pretrained data!")

    # in each run, randomly sample a batch of data from the training dataset
    for eachbatch in range(args.start_step,args.total_iter):
        #################################### Generate noise sequence ##################################################
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))

        ################################### Pretraining state ##############################################
        if eachbatch < args.core * 20000:
           snr1=3* (1-eachbatch/(args.core * 20000))+ (eachbatch/(args.core * 20000)) * args.snr1
           snr2 = 100
           belief_threshold = 0.9+0.09999*(eachbatch/(args.core * 20000))
        elif eachbatch < args.core * 40000:
           snr2= 100 * (1-(eachbatch-args.core * 20000)/(args.core * 20000))+ ((eachbatch-args.core * 20000)/(args.core * 20000)) * args.snr2
           snr1=args.snr1
           belief_threshold = 0.99999+0.0000099*((eachbatch-args.core * 20000)/(args.core * 20000))
        ################################### Finetuning state ##############################################
        else:
           belief_threshold = args.belief_threshold
           snr2=args.snr2
           snr1=args.snr1
        ####################################Set forward noise and feedback noise#############################################
        std1 = 10 ** (-snr1 * 1.0 / 10 / 2) #forward snr
        std2 = 10 ** (-snr2 * 1.0 / 10 / 2) #feedback snr
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        ################################## Simulate multicores by singlecore ###############################################
        if np.mod(eachbatch, args.core) == 0:
            w_locals = []
            w0 = model.state_dict()
            w0 = copy.deepcopy(w0)
        else:
            # Use the common model to have a large batch strategy
            model.load_state_dict(w0)
        
        ################################## Training ###############################################
        if args.multclass:
           bVec_mc = torch.matmul(bVec,map_vec)
           ys = bVec_mc.long().contiguous().view(-1)
        else:
           ys = bVec.long().contiguous().view(-1)
        train_log,preds,losses = model(belief_threshold,
                            eachbatch, 
                            bVec.to(args.device), 
                            fwd_noise_par.to(args.device),
                            fb_noise_par.to(args.device),
                            ys,
                            isTraining=1)
        # Save the model
        w1 = model.state_dict()
        w_locals.append(copy.deepcopy(w1))
        ###################### untill core number of iterations are completed ####################
        if np.mod(eachbatch, args.core) != args.core - 1:
            continue
        else:
            ########### When core number of models are obtained #####################
            w2 = ModelAvg(w_locals)  # Average the models
            model.load_state_dict(copy.deepcopy(w2))
            ##################### change the learning rate ##########################
            if args.use_lr_schedule:
                args.scheduler.step()
        ################################ Observe test accuracy##############################
        with torch.no_grad():
            decodeds = preds.max(dim=1)[1]
            succRate = sum(decodeds == ys.to(args.device)) / len(ys)
            log = {"batch":eachbatch,
                   "snr1":args.snr1,
                   "snr2":args.snr2,
                   "lr":args.optimizer.state_dict()['param_groups'][0]['lr'],
                   "BER":1 - succRate.item(),
                   "num":sum(decodeds != ys.to(args.device)).item(),
                   "final_loss":train_log[-1]['loss'],
                   "losses":losses,
                   "train_log":train_log}
            print(log)
        #############################Save Model###########################
        state_dict = {
                    'model': model.state_dict(),
                    'optimizer': args.optimizer.state_dict(),
                    'lr_scheduler': args.scheduler.state_dict(),
                    'epoch': eachbatch,
                    }
        if np.mod(eachbatch, args.core * 20000) == args.core - 1 and eachbatch >= 40000:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            saveDir = 'weights/model_weights_{}_{}_'.format(snr1,snr2) + str(eachbatch)
            torch.save(state_dict, saveDir)
        else:
            if not os.path.exists('weights'):
                os.mkdir('weights')
            torch.save(state_dict, 'weights/latest')


def evaluate_model(model, args):
    checkpoint = torch.load(args.test_model)
    # # ======================================================= load weights
    model.load_state_dict(checkpoint['model'])
    print(args)
    print("-->-->-->-->-->-->-->-->-->--> start testing ...")
    model.eval()
    map_vec = torch.tensor([1,2,4])
    args.numTestbatch = 100000
    bitErrors = 0
    pktErrors = 0

    for eachbatch in range(args.numTestbatch):
        bVec = torch.randint(0, 2, (args.batchSize, args.numb_block, args.block_size))
        std1 = 10 ** (-args.snr1 * 1.0 / 10 / 2)
        std2 = 10 ** (-args.snr2 * 1.0 / 10 / 2)
        fwd_noise_par = torch.normal(0, std=std1,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        fb_noise_par = torch.normal(0, std=std2,
                                     size=(args.batchSize, args.numb_block, args.truncated),
                                     requires_grad=False)
        if args.snr2 == 100:
            fb_noise_par = 0* fb_noise_par
        if args.multclass:
            bVec_mc = torch.matmul(bVec,map_vec)
            ys = bVec_mc.long().contiguous().view(-1)
        else:
            ys = bVec.long().contiguous().view(-1)
        with torch.no_grad():
            test_log,preds = model(args.belief_threshold, eachbatch, bVec.to(args.device), fwd_noise_par.to(args.device),fb_noise_par.to(args.device), ys,isTraining=0)
            avg_codelen = compute_avgcodelength(test_log)

            preds1 =  preds.contiguous().view(-1, preds.size(-1))
            decodeds = preds1.max(dim=1)[1]
            decisions = decodeds != ys.to(args.device)
            bitErrors += decisions.sum()
            BER = bitErrors / (eachbatch + 1) / args.batchSize / args.numb_block
            pktErrors += decisions.view(args.batchSize, args.numb_block).sum(1).count_nonzero()
            PER = pktErrors / (eachbatch + 1) / args.batchSize
            log = {"batch":eachbatch,
                   "avg_codelen":avg_codelen,
                   "snr1":args.snr1,
                   "snr2":args.snr2,
                   "BER":BER.item(),
                   "bitErrors":bitErrors.item(),
                   "PER":PER.item(),
                   "num":sum(decodeds != ys.to(args.device)).item(),
                   "test_log":test_log}
            print(log)
    BER = bitErrors.cpu() / (args.numTestbatch * args.batchSize * args.K)
    PER = pktErrors.cpu() / (args.numTestbatch * args.batchSize)
    print(BER)
    print(PER)
    print("Final test BER = ", torch.mean(BER).item())
    print("Final test PER = ", torch.mean(PER).item())
    pdb.set_trace()

if __name__ == '__main__':
    # ======================================================= parse args
    args = args_parser()
    restriction_dict = {'min':0.999,
                        'low':0.9999,
                        'mid':0.99999,
                        'high':0.999999,
                        'max':0.9999999,}
    args.belief_threshold = restriction_dict[args.restriction]
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ########### path for saving model checkpoints ################################
    args.saveDir = 'weights/model_weights'  # path to be saved to
    ################## Model size part ###########################################
    args.d_model_trx = args.heads_trx * args.d_k_trx # total number of features
    ##############################################################################
    args.total_iter = 10000 * args.totalbatch + 1 + args.core
    # ======================================================= Initialize the model
    model = DeepVLF(args).to(args.device)
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
    # ======================================================= run
    if args.train == 1:
        train_model(model, args)
        args.test_model = 'weights/latest'
        args.batchSize = 100000
        evaluate_model(model, args)
    else:
        evaluate_model(model, args)
