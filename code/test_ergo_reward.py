import argparse
from reward import Reward
from config import AMINO_ACIDS
import pdb
import warnings
import numpy as np
import os
import copy
path = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ergo_model_file", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--pair_path", type=str, default=path + "/ERGO/data/McPAS-TCR/selected_pairs.txt")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_method", type=str, default="blosum")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=50000)
    
    parser.add_argument("--perturb", action="store_true")
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--output_true", action="store_true")
    
    parser.add_argument("--clip_norm", type=float, default=50.0)
    parser.add_argument("--load_epoch", type=int, default=10000)
    args = parser.parse_args()
    
    embed_methods = ['blosum', 'onehot', 'deep']
    
    embedding_size = 0

    reward_model = Reward(0.5, 0.5, ergo_model_file=args.ergo_model_file)
    
    train_pairs = [line.strip().split() for line in open(args.pair_path, 'r').readlines() if len(line.split()[0]) < 28]
    
    np.random.seed(1000)
    train_pairs_idx = np.random.randint(0, high=len(train_pairs)-1, size=10000)
    train_pairs = [train_pairs[idx] for idx in train_pairs_idx]
    
    output_file = open(args.output, 'w')
    
    for idx in range(0, len(train_pairs), args.batch_size):
        batch_pairs = [train_pairs[i] for i in range(idx, min(len(train_pairs), idx + args.batch_size))]
        batch_tcrs = [pair[0] for pair in batch_pairs]
        batch_peptides = [pair[1] for pair in batch_pairs]
        
        scores = reward_model.get_ergo_reward( batch_tcrs, batch_peptides )
        
        output_str = ""
        for i in range(len(batch_pairs)):
            output_str += "%s %s %.4f\n" % (batch_pairs[i][1], batch_pairs[i][0], scores[i])
            
        output_file.write(output_str)
    
    output_file.close()
