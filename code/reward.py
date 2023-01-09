import warnings
from functools import wraps

from sklearn.mixture import GaussianMixture
import numpy as np
import argparse
import pickle

import sys, os
cwd = os.path.dirname(os.path.abspath(__file__))
ERGO_path = cwd + "/ERGO/"
reward_path = cwd + "/reward/"
sys.path.append(ERGO_path)
sys.path.append(reward_path)

import lstm_utils as lstm
import torch
from AE import AE
from ERGO_models import DoubleLSTMClassifier
from ERGO_models import AutoencoderLSTMClassifier
import ae_utils as ae
import pdb
from tcr_lstm import TCRLSTM
import copy

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
amino_to_ix = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}

ae_file = ERGO_path + 'TCR_Autoencoder/tcr_ae_dim_100.pt'
pep_atox = {amino: index for index, amino in enumerate(['PAD'] + amino_acids)}
tcr_atox = {amino: index for index, amino in enumerate(amino_acids + ['X'])}


def ignore_warnings(f):
    @wraps(f)
    def inner(*args, **kwargs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("ignore")
            response = f(*args, **kwargs)
        return response
    return inner

class Reward:
    def __init__(self, beta, criteria, use_gmm=True, seman_path=cwd+"/like_ratio/semantic_model", back_path=cwd+"/like_ratio/background_model",\
                       ae_model_file=cwd + "/reward/ae_model", gmm_model_file=cwd + "/reward/gmm.pkl", ergo_model_file=ERGO_path + "models/ae_mcpas1.pt"):
        self.ergo_model_file = ergo_model_file
        self.ergo_model = self.__init_ergo_model(ergo_model_file)
        
        self.use_gmm = use_gmm
        if use_gmm:
            self.ae_model = self.__init_ae_model(ae_model_file)
            self.gmm_model = self.__init_gmm_model(gmm_model_file)
        else:
            self.semantics_model, self.background_model = self.__init_likelihood_model(seman_path, back_path)
        
        self.max_len = 28
        self.beta = beta
        self.criteria = criteria
        
    def __init_ergo_model(self, ergo_model_file):
        if "lstm" in ergo_model_file:
            model = DoubleLSTMClassifier(10, 500, 0.1, device)
        else:
            model = AutoencoderLSTMClassifier(10, device, 28, 21, 100, 1, ae_file, False)
        
        checkpoint = torch.load(ergo_model_file, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model

    def __init_likelihood_model(self, seman_path, back_path):
        semantics_model = TCRLSTM(20, 64, 8, \
                         'blosum', beta=1)
        semantics_model.load_state_dict(torch.load( seman_path, map_location=torch.device(device)))
    
        background_model = TCRLSTM(20, 64, 8, \
                         'blosum', beta=1)
        background_model.load_state_dict(torch.load( back_path, map_location=torch.device(device)))
        return semantics_model, background_model

    def __init_ae_model(self, ae_model_file):
        model = AE(20, 64, 16, 'blosum')
        model.load_state_dict(torch.load(ae_model_file, map_location=torch.device(device)))
        return model

    def __init_gmm_model(self, gmm_model_file):
        with open(gmm_model_file, 'rb') as f:
            model = pickle.load(f)
        return model

    def __get_ergo_preds(self, tcrs, peps):
        signs = [0] * len(tcrs)
        
        if "ae" in self.ergo_model_file:
            test_batches = ae.get_full_batches(tcrs, peps, signs, tcr_atox, pep_atox, 1, self.max_len)
            preds = ae.predict(self.ergo_model, test_batches, device)
        else:
            _tcrs, _peps = lstm.convert_data(tcrs, peps, amino_to_ix)    
            test_batches = lstm.get_full_batches(_tcrs, _peps, signs, 1, amino_to_ix)
            preds = lstm.predict(self.ergo_model, test_batches, device)
        
        return preds
    
    @ignore_warnings
    def reward(self, tcrs, peptides):
        
        scores = self.__get_ergo_preds(copy.deepcopy(tcrs), peptides)
        
        all_rewards = None
        if self.use_gmm:
            seq_edit_acc, likelihoods = self.get_gmm_reward(tcrs)
            rewards = np.clip( (seq_edit_acc + likelihoods - self.criteria), a_min=None, a_max=0)
            all_rewards = np.stack( (seq_edit_acc, likelihoods), axis=1)
            
        else:
            seman_like, back_like = self.get_like_reward(tcrs)
            rewards = np.clip( (seman_like - back_like - self.criteria), a_min=None, a_max=0)
            all_rewards = np.stack( (seman_like, back_like), axis=1)
        
        final_rewards = scores + self.beta * rewards
        rewards = np.stack( (final_rewards, scores, all_rewards[:, 0], all_rewards[:, 1]), axis=1)
        
        return rewards

    @ignore_warnings
    def reward2(self, tcrs, peptides):
        seqs, seq_edit_dists, z = self.ae_model.edit_dist(tcrs)
        likelihoods = self.gmm_model.score_samples(z)
        
        tcr_reward = (1-seq_edit_dists) + np.exp((likelihoods+10) / 10)
        final_rewards = self.beta * np.clip( tcr_reward, a_min=None, a_max=self.criteria)
        scores = np.zeros( (len(tcrs)) )
        indices = np.where(tcr_reward >= self.criteria)[0]
        if indices.shape[0] > 0: 
            selected_tcrs = [tcrs[idx] for idx in indices]
            selected_peptides = [peptides[idx] for idx in indices]
            selected_scores = self.__get_ergo_preds(selected_tcrs, selected_peptides)
            for i, indice in enumerate(indices):
                final_rewards[indice] += selected_scores[i]
                scores[indice] = selected_scores[i]
        
        rewards = np.stack( (final_rewards, scores, (1-seq_edit_dists), np.exp((likelihoods+10)/10)), axis=1)
        return rewards
        
    def get_gmm_reward(self, tcrs):
        seqs, seq_edit_dists, z = self.ae_model.edit_dist(tcrs)
        likelihoods = self.gmm_model.score_samples(z)
        
        return (1-seq_edit_dists), np.exp((likelihoods + 10)/10)

    def get_like_reward(self, tcrs):
        sm_likelihood = self.semantics_model.test(tcrs).detach().cpu().numpy()
        bg_likelihood = self.background_model.test(tcrs).detach().cpu().numpy()
        
        return sm_likelihood, bg_likelihood

    def get_ergo_reward(self, tcrs, peptides):
        return self.__get_ergo_preds(tcrs, peptides)
