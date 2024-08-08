import argparse

device = 'cuda:0'

K = 6
m = 6
ell = int(K/m)
memory = K

T = 18

if T == 9:
	snr1 = -1.0
elif T == 8:
	snr1 = 0.0
elif T == 7:
	snr1 = 1.0
elif T == 6:
	snr1 = 2.0
elif T == 5:
    snr1 = 3.0

# snr1 = snr1 - 1.0
snr1 = 1.0
snr2 = 20.0
# # snr1 = 0.0

train = 1

seed = 123

batchSize = int(50000)

def args_parser():
    parser = argparse.ArgumentParser()

    # Sequence arguments
    parser.add_argument('--snr1', type=float, default= snr1, help="Transmission SNR")
    parser.add_argument('--snr2', type=float, default= snr2, help="Feedback SNR")
    parser.add_argument('--K', type=int, default=K, help="Sequence length")
    parser.add_argument('--m', type=int, default=m, help="Block size")
    parser.add_argument('--ell', type=int, default=ell, help="Number of bit blocks")
    parser.add_argument('--T', type=int, default=T, help="Number of interactions")
    parser.add_argument('--seq_reloc', type=int, default=1)
    parser.add_argument('--memory', type=int, default=K)
    parser.add_argument('--core', type=int, default=1)
    parser.add_argument('--NS_model', type=int, default=2)
    parser.add_argument('--seed', type=int, default=seed)

    # Transformer arguments
    parser.add_argument('--heads_trx', type=int, default=1, help="number of heads for the multi-head attention")
    parser.add_argument('--d_k_trx', type=int, default=32, help="number of features for each head")
    parser.add_argument('--N_trx', type=int, default=2, help=" number of layers in the encoder and decoder")
    parser.add_argument('--dropout', type=float, default=0.0, help="prob of dropout")
    parser.add_argument('--custom_attn', type=bool, default = True, help= "use custom attention")
    parser.add_argument('--vv', type=int, default=1)

    # Learning arguments
    parser.add_argument('--load_weights') # None
    parser.add_argument('--train', type=int, default= train)
    parser.add_argument('--reloc', type=int, default=1, help="w/ or w/o power rellocation")
    parser.add_argument('--totalbatch', type=int, default=120101, help="number of total batches to train")
    parser.add_argument('--batchSize', type=int, default=batchSize, help="batch size")
    parser.add_argument('--opt_method', type=str, default='adamW', help="Optimization method adamW,lamb,adam")
    parser.add_argument('--clip_th', type=float, default=0.5, help="clipping threshold")
    parser.add_argument('--use_lr_schedule', type=bool, default = True, help="lr scheduling")
    parser.add_argument('--multclass', type=bool, default = True, help="bit-wise or class-wise training")
    parser.add_argument('--embedding', type=bool, default = False, help="vector embedding option")
    parser.add_argument('--embed_normalize', type=bool, default = True, help="normalize embedding")
    parser.add_argument('--belief_modulate', type=bool, default = True, help="modulate belief [-1,1]")
    parser.add_argument('--clas', type=int, default = 8, help="number of possible class for a block of bits")
    parser.add_argument('--rev_iter', type=int, default = 0, help="number of successive iteration")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--wd', type=float, default=0.01, help="weight decay")
    parser.add_argument('--device', type=str, default=device, help="GPU")
    args = parser.parse_args()

    return args