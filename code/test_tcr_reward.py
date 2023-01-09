import argparse
from reward import Reward
from config import AMINO_ACIDS
import warnings
import numpy as np
import random
import os
from functools import partial
from multiprocessing import Pool
import copy
path = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=UserWarning)

def perturb_TCR(tcr, num=1):
    while True:
        selected_poss = np.random.randint(0, high=len(tcr), size=num)
        if len(np.unique(selected_poss)) == num: break
    
    new_tcr = tcr
    for pos in selected_poss:
        while True:
            amino = random.choice( AMINO_ACIDS )
            if amino == new_tcr[pos]: continue
            tmp_tcr = new_tcr[:pos] + amino
            if pos != len(tcr) - 1: tmp_tcr += new_tcr[pos+1:]
            new_tcr = tmp_tcr
            break
        
    return new_tcr
    
    
def perturb_TCRs2(batch_TCRs, perturb_num, ncpu=10):
    #perturb_TCR(batch_TCRs[0], num=perturb_num)
    func = partial( perturb_TCR, num=perturb_num )
    with Pool(processes=10) as pool:
        result = pool.map(func, batch_TCRs)
    
    return result
    
def perturb_TCRs(batch_TCRs, batch_mats, rate=0.2):
    mute_pos = np.where(batch_mats<rate)
    mute_ac = np.random.choice(AMINO_ACIDS, mute_pos[0].shape[0])
    
    new_batch_TCRs = copy.deepcopy(batch_TCRs)
    for i in range(mute_pos[0].shape[0]):
        pos_i = mute_pos[0][i]
        pos_j = mute_pos[1][i]
        
        tcr = new_batch_TCRs[pos_i]
        if len(tcr) <= pos_j: continue
        
        tcr = tcr[:pos_j] + mute_ac[i] + tcr[pos_j+1:]
        new_batch_TCRs[pos_i] = tcr
        
    return new_batch_TCRs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/tcrdb/test_uniq_tcr_seqs.txt")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_method", type=str, default="blosum")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=10000)
    
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--use_rate", action="store_true")
    parser.add_argument("--perturb_rate", type=float, default=0.1)
    parser.add_argument("--perturb_num", type=int, default=1)
    parser.add_argument("--output_true", action="store_true")
    
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--load_epoch", type=int, default=10000)
    args = parser.parse_args()
    
    embed_methods = ['blosum', 'onehot', 'deep']
    
    embedding_size = 0

    reward_model = Reward(0.5, 0.5)
    
    #tcrs = [tmp[0] for tmp in tcr_pep]
    #peps = [tmp[1] for tmp in tcr_pep]

    #edit_dist, gmm_lh = reward_model.get_gmm_reward(tcrs)


    test_TCRs = [line.strip() for line in open(args.data_path, 'r').readlines()]
    max_len = max([len(TCR) for TCR in test_TCRs])
    
    #print("with %s device" % (device))
    
    beta = args.beta
    if args.output_true: true_output_file = open("%s/true_output.txt" % (path), 'w')
    
    if args.use_rate:
        perturb_output_file = open("%s/perturb_output_%.2f.txt" % (path, args.perturb_rate), 'w')
    else:
        perturb_output_file = open("%s/perturb_output_%d.txt" % (path, args.perturb_num), 'w')
    
    test_num = 0
    
    for idx in range(0, len(test_TCRs), args.batch_size):
        test_num += args.batch_size
        batch_TCRs = test_TCRs[idx:min(idx + args.batch_size, len(test_TCRs))]
        
        
        if args.use_rate:
            if args.perturb_rate == 1:
                perturbed_TCRs = []
                for TCR in batch_TCRs:
                    tmp = "".join(list(np.random.choice(AMINO_ACIDS, len(TCR) - 2)))
                    
                    new_TCR  = "C" + tmp + "F"
                    perturbed_TCRs.append(new_TCR)
            else:
                perturb_rates = np.random.rand( len(batch_TCRs), max_len )
                perturbed_TCRs = perturb_TCRs( batch_TCRs, perturb_rates, rate=args.perturb_rate )
        else:
            perturbed_TCRs = perturb_TCRs2( batch_TCRs, args.perturb_num )
        
        if args.output_true:
            true_edit_dist, true_gmm_lh = reward_model.get_gmm_reward(batch_TCRs)
        perturb_edit_dist, perturb_gmm_lh = reward_model.get_gmm_reward(perturbed_TCRs)
        
        true_str, perturb_str = "", ""
        for i in range(len(batch_TCRs)):
            if args.output_true: true_str += "%s %.4f %.4f\n" % (batch_TCRs[i], true_edit_dist[i], true_gmm_lh[i])
            perturb_str += "%s %s %.4f %.4f\n" % (perturbed_TCRs[i], batch_TCRs[i], perturb_edit_dist[i], perturb_gmm_lh[i])
            
        if args.output_true: true_output_file.write(true_str)
        perturb_output_file.write(perturb_str)

    if args.output_true: true_output_file.close()
    perturb_output_file.close()
