import os
import torch
#from stable_baselines3.common.utils import get_device

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cuda"
cwd = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.dirname(cwd)

TCR_LENGTH = 27
AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
PEP_LENGTH = 25
BLOSUM = cwd + "/blosum.txt"

LEARNED_DIM = 20

TCR_FILE = os.path.dirname(cwd) + "/data/tcrdb/train_uniq_tcr_seqs.txt"
LENGTH_DIST_FILE = os.path.dirname(cwd) + "/data/tcrdb/length_dist.txt"
LENGTH_DIST = {}
for line in open(LENGTH_DIST_FILE, 'r'):
    length, freq = line.strip().split()
    LENGTH_DIST[length] = freq
