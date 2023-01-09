from sklearn.mixture import GaussianMixture
import numpy as np
import argparse
import pickle

import sys, os
sys.path.append(os.path.abspath(__file__))
import torch
from AE import AE
import pdb
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--ncomp", type=int, default=2)
    parser.add_argument("--model", type=str)
    parser.add_argument("--save_path", type=str)
    
    args = parser.parse_args()

    data = [line.strip() for line in open(args.data, 'r').readlines()][:1000000]
    
    model = AE(20, 64, 16, 'blosum')
    model.load_state_dict(torch.load(args.model, map_location=torch.device(device)))
    #seq = model.generate(model.encode([data[0]]))
    #pdb.set_trace()
    latent_codes = (model.encode(data)).detach().numpy()
    
    gm = GaussianMixture(n_components=args.ncomp, random_state=0).fit(latent_codes)
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(gm, f)
    
