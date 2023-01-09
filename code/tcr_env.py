import argparse
import torch
import gym
import itertools
import numpy as np
import copy
import random
import time
import csv
from contextlib import contextmanager
from config import AMINO_ACIDS, TCR_LENGTH, PEP_LENGTH, LENGTH_DIST, TCR_FILE
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_util import make_vec_env
from data_utils import num2seq, seq2num, n2a_func
cwd = os.path.dirname(os.path.abspath(__file__))

from reward import Reward


class TCREnv(gym.Env):
    def __init__(self,
        action_space: gym.spaces,
        observation_space: gym.spaces,
        args: dict,
        max_pep_len: int = 25,
        max_tcr_len: int = 27,
    ):
        super(TCREnv, self).__init__()
        self.possible_amino_types = np.array(AMINO_ACIDS, dtype=object)
        self.use_gmm = args.use_gmm
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.allow_imm_rew = args.allow_imm_rew
        self.rate = args.rate
        self.terminal = args.terminal
        self.reward_type = args.reward_type
        self.reward = np.zeros(4)    
        
        self.gmm_stop_criteria = args.gmm_stop_criteria
        self.score_stop_criteria = args.score_stop_criteria
        
        self.use_step = args.use_step
        self.max_step = args.max_step
        self.final_rew = args.allow_final_rew
        self.no_mod_penalty = args.no_mod_penalty
        self.mod_neg_penalty = args.mod_neg_penalty
        self.mod_pos_penalty = args.mod_pos_penalty
        self.discount_penalty = args.discount_penalty
        
        self.anneal_nomod_step = args.anneal_nomod_step
        self.anneal_nomod_rate = args.anneal_nomod_rate
        
        peptide_path = args.peptide_path
        
        self.peptides = set()
        for line in open(peptide_path, 'r').readlines():
            elem = line.strip()
            if len(elem) > 25: continue
            if "X" in elem: continue
            if "\\" in elem: continue
            
            self.peptides.add(elem)
        
        self.peptides = list(self.peptides)
        
        self.len_step = 0    
        self.max_tcr_len = max_tcr_len
        self.max_pep_len = max_pep_len
        
        
        self.init_peptide = None
        self.num_step = 0
        self.tcr_seqs = [line.strip() for line in open(TCR_FILE, 'r').readlines() if len(line.strip()) <= self.max_tcr_len]
        
        self.tcr_len, self.tcr_len_freqs, self.tcr_amino_dist = self.__get_tcr_dist(self.tcr_seqs)
        print("finish build")

    
    def __get_tcr_dist(self, tcr_seqs):
        tcr_seqs = [tcr_seq for tcr_seq in tcr_seqs if len(tcr_seq) <= self.max_tcr_len]
        
        lengths = [len(seq) for seq in tcr_seqs]
        unique, counts = np.unique(lengths, return_counts=True)
        
        freqs = counts / np.sum(counts)
        
        dicts = {}
        seq_arrays, _ = seq2num(tcr_seqs, max_len=self.max_tcr_len)
        seq_arrays = seq_arrays.astype(int)-1
        
        for length in unique:
            dicts[length] = np.zeros((length, len(AMINO_ACIDS)))
            idxs = np.where(np.array(lengths) == length)[0]
            array = np.take(seq_arrays, idxs, 0)[:, :length]
            
            for i in range(length):
                pos_unique, pos_counts = np.unique(array[:, i], return_counts=True)
            
                for j, amino in enumerate(pos_unique): dicts[length][i, amino] = pos_counts[j] / len(idxs)
        
            
        return unique, freqs, dicts
        
    def reset_with_peptide(self, peptide):
        self.needs_reset = False
        
        tcr = self.init_tcr()

        tcrs = seq2num([tcr], max_len=self.max_tcr_len)
        peptides = seq2num([peptide], max_len=self.max_pep_len)
        
        self.len_step = 0
        
        if self.use_step:
            self.state = torch.cat((tcr, peptide, torch.LongTensor([[self.len_step]])), dim=1)
        else:
            self.state = torch.cat((tcr, peptide), dim=1)
        
        return self.state.squeeze(0)
        
    def init_tcr(self):
        tcr_seq = random.choice(self.tcr_seqs)
        self.initial_tcr = tcr_seq
        return tcr_seq

    def reset(self, peptide=None, init_tcr=None):
        if peptide is None:
            idx = np.random.choice(len(self.peptides), 1)
            peptides = [self.peptides[i] for i in idx]
            self.peptide = peptides[0]
        else:
            self.peptide = peptide
        
        
        ratio = random.random()
        
        original_peptides = []
        if init_tcr is None:
            tcr = self.init_tcr()
        else:
            self.initial_tcr = tcr = init_tcr
            
        tcrs, length = seq2num([tcr], max_len=self.max_tcr_len)
        peptides, length = seq2num([self.peptide], max_len=self.max_pep_len)
        tcrs = torch.LongTensor(tcrs)
        peptides = torch.LongTensor(peptides)
         
        self.len_step = 0
        
        if self.use_step:
            self.state = torch.cat((tcrs, peptides, torch.LongTensor([[self.len_step]])), dim=1)
        else:
            self.state = torch.cat((tcrs, peptides), dim=1)
        
        return self.state.squeeze(0)
    
    def check_terminal(self, final_reward, score, dist, likelihood, actions=None):
        if (score >= self.score_stop_criteria and dist + likelihood >= self.gmm_stop_criteria) or self.len_step >= self.max_step:
            return True
        else:
            return False

    def _get_reward(self, tcrs, peptides, actions=None):
        final_reward, score, dist, likelihood = self.reward
        
        self.num_step += 1
        
        terminal = self.check_terminal(final_reward, score, dist, likelihood, actions=actions)
                     
        reward = 0
        
        if actions is None:
            reward = 0
        elif not (self.final_rew and terminal) and (self.allow_imm_rew is None or self.num_step < self.allow_imm_rew): 
            reward = min(dist + likelihood - self.gmm_stop_criteria, 0)
        elif terminal:
            reward = final_reward
            
        return reward, score, dist, likelihood, terminal
    
    def _edit_sequence(self, peptides, actions):
        new_peptides = []
        for i in range(len(peptides)):
            peptide = peptides[i]
            action = actions[i, :]
            
            position = action[0]
            new_peptide = ""
            
            new_peptide = peptide[:position] + AMINO_ACIDS[action[1]-1]
                
            if position < len(peptide) - 1:
                new_peptide += peptide[position+1:]
                
            new_peptides.append(new_peptide)
            
        return new_peptides
    
    def step(self, actions: torch.Tensor, reward: np.ndarray):
        
        tcrs = num2seq(self.state[:, :27])
        peptides = num2seq(self.state[:, 27:53])
        self.reward = reward
        ### take action
        if len(actions.shape) == 1: actions = np.expand_dims(actions, axis=0)
        new_tcrs = self._edit_sequence(tcrs, actions)
         
        self.len_step += 1
        term_reward, score, reward1, reward2, terminal = self._get_reward(new_tcrs, peptides, actions=actions)
        
        info = {}
        info['terminal'] = str(terminal)
        info['action'] = ",".join([str(actions[0][i]) for i in range(2)])
        info['old_tcr'] = tcrs[0]
        info['new_tcr'] = new_tcrs[0]
        info['init_tcr'] = self.initial_tcr if self.initial_tcr is not None else "None"
        info['peptide'] = peptides[0]
        info['rewards'] = float(term_reward)
        info['score'] = float(score)
        
        if self.use_gmm:
            info['score1'] = float(reward1)
            info['score2'] = float(reward2)
        else:
            info['score1'] = float(reward1)
            info['score2'] = float(reward2)
        
        if len(new_tcrs[0]) > self.max_tcr_len: print("!!%s ; %s ; %s; %s" % (info['action'], info['old_tcr'], info['new_tcr'], info['init_tcr']))
        new_tcr_vecs = torch.LongTensor(seq2num(new_tcrs, max_len=self.max_tcr_len)[0])
        self.state = torch.cat((new_tcr_vecs, self.state[:, 27:]), dim=1)
        
        return self.state.squeeze(0), term_reward, terminal, info

if __name__ == '__main__':
    import sys, os
    from ppo import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from subproc_vec_env import SubprocVecEnv
    from policy import PolicyNet
    from seq_embed import SeqEmbed
    import pickle
    import torch
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument('--peptide_path', type=str)
    parser.add_argument('--path', type=str, help="path to save results")
    parser.add_argument('--peptide_name', type=str, default="class1_pseudosequences.csv", help="path of peptides") 
    
    # reward design
    parser.add_argument('--reward_type', type=str, default="game", help="select game or molecule")
    parser.add_argument('--terminal', action="store_true", help="whether using the no-modification action as termination")
    parser.add_argument('--discount_penalty', type=float, default=0.8, help="used for molecule modification-based reward design")
    
    parser.add_argument('--mod_pos_penalty', type=float, default=1, help="penalty for each step")
    parser.add_argument('--no_mod_penalty', type=float, default=-0.5, help="penalty for no modification")
    parser.add_argument('--mod_neg_penalty', type=float, default=-1, help="penalty for negative modification")
    
    parser.add_argument('--allow_imm_rew', type=int, default=0, help="whether use immediate reward or not: (None represent using imm reward; 0 represent not using imm reward")
    parser.add_argument('--allow_final_rew', action="store_false", help="whether use final reward or not")

    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--rate', type=float, default=10, help="weight of final reward")
    
    parser.add_argument('--anneal_nomod_step', type=int, default=10000)
    parser.add_argument('--anneal_nomod_rate', type=float, default=0.05)
    parser.add_argument('--encode', type=str, default="deep_blosum_onehot", help="amino acid encoding")
    
    # parameters for policy network
    parser.add_argument('--peptide_kmer', type=int, default=3, help="need to be specified when using CNN to encode alleles")
    parser.add_argument('--embed_peptide', type=str, default="LSTM", help="model architecture used to encode alleles")
    
    parser.add_argument('--hidden_dim', type=int, default=128, help="dimension of shared layer in policy network and value network")
    parser.add_argument('--latent_dim', type=int, default=64, help="dimension of latent layer in two networks")

    # ppo algorithms
    parser.add_argument('--gamma', type=float, default=0.99, help="discount_factor")
    parser.add_argument('--steps', type=int, default=10000000, help="total time steps")
    parser.add_argument('--ent_coef', type=float, default=0.01, help="encourage exploration")
    
    parser.add_argument('--clip', type=float, default=0.2, help="")
    parser.add_argument('--kl_target', type=float, default=0.01, help="")
    parser.add_argument('--max_len', type=int, default=27)
    
    # environment
    parser.add_argument('--ergo_model', type=str, default=cwd + "/ERGO/models/ae_mcpas1.pt")
    parser.add_argument('--use_step', action="store_true")
    parser.add_argument('--use_gmm', action="store_false")
    parser.add_argument('--score_stop_criteria', type=float, default=0.9, help="stop_criteria")
    parser.add_argument('--gmm_stop_criteria', type=float, default=1.2577, help="stop_criteria")
    parser.add_argument('--num_envs', type=int, default=20, help="number of environments")
    parser.add_argument('--n_steps', type=int, default=256, help="number of roll out steps")
    parser.add_argument('--max_step', type=int, default=8, help="maximum number of steps")
    
    parser.add_argument('--bad_ratio', type=float, default=0.5) 
    parser.add_argument('--rate_for_bad_ratio', type=float, default=5.0)
    
    args = parser.parse_args()

    t1 = time.time()
    dir_name = ""
    names = ['beta', 'bad_ratio', 'gamma', 'n_steps', 'embed_latent_dim']
    
    for name in names:
        attr = None
        for arg in vars(args):
            if arg != name: continue
            attr = getattr(args, arg)
            break
        
        if isinstance(attr, str):
            if "/" in attr: attr = attr.split("/")[-1]
            dir_name += attr+"_"
        else:
            dir_name += str(attr)+"_"
    
    if "mcpas" in args.ergo_model:
        dir_name = "mcpas_" + dir_name
    
    if "vdjdb" in args.ergo_model:
        dir_name = "vdjdb_" + dir_name
    
    path = args.path + dir_name[:-1]
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    action_space = gym.spaces.multi_discrete.MultiDiscrete([args.max_len, 20])
    if args.use_step:
        observation_space = gym.spaces.MultiDiscrete([20] * (26 + args.max_len))
    else:
        observation_space = gym.spaces.MultiDiscrete([20] * (25 + args.max_len))
    
    reward_model = Reward(args.beta, args.gmm_stop_criteria, use_gmm=args.use_gmm, ergo_model_file=args.ergo_model)
    
    m_env_kwargs = {"action_space":action_space, "observation_space":observation_space, \
                    "args": args, "max_tcr_len": args.max_len}
  
    m_env = make_vec_env(TCREnv, n_envs=args.num_envs, env_kwargs=m_env_kwargs, vec_env_cls=SubprocVecEnv)
    
    if_deep = True if "deep" in args.encode else False
    if_blosum = True if "blosum" in args.encode else False
    if_onehot = True if "onehot" in args.encode else False
    ftype = {'deep':if_deep, 'blosum': if_blosum, 'onehot': if_onehot}
    embed_dim = sum([1 for boo in ftype.values() if boo]) * 20
    
    config = {"ftype":ftype, "max_tcr_len": args.max_len, \
              "hidden_dim": args.hidden_dim, "peptide_kmer": args.peptide_kmer, \
              "embed_peptide": args.embed_peptide, "use_step": args.use_step}
    
    seq_features = SeqEmbed(config)
     
    policy_kwargs = dict(features_extractor=seq_features, \
                        net_arch = [args.hidden_dim, dict(vf=[args.latent_dim], pi=[args.latent_dim])], \
                        use_step = args.use_step, max_tcr_len = args.max_len)
    
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=path+'/', name_prefix='rl_model')
    
    buffer_config = dict(max_len = 5000, init_size = 1000, bad_ratio=args.bad_ratio, rate_for_bad_ratio=args.rate_for_bad_ratio)
    
    model = PPO(PolicyNet, m_env, verbose=1, reward_model=reward_model, n_steps=args.n_steps, ent_coef=args.ent_coef, gamma=args.gamma, clip_range=args.clip, target_kl=args.kl_target, buffer_config=buffer_config, policy_kwargs=policy_kwargs)
    
    model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
    t2 = time.time()

    print("finish training in %.4f" % (t2 -t1))
    print("saving model.....")
    model.save(path+"/ppo_tcr")
