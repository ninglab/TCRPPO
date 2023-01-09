import os
import pdb
import torch
import torch.nn as nn
import numpy as np
from config import AMINO_ACIDS, BLOSUM, TCR_LENGTH, PEP_LENGTH, LEARNED_DIM, device

class SeqEmbed(nn.Module):
    def __init__(self, config):
        super(SeqEmbed, self).__init__()        
        self.hidden_dim = config['hidden_dim']
        self.kmer = config["peptide_kmer"]
        self.use_step = config["use_step"]
          
        self.features_dim = self.hidden_dim * 4
        self.max_tcr_len = config["max_tcr_len"]   
        # encode features
        self.deep, self.blosum, self.onehot = False, False, False
        ftype = config['ftype']
        
        self.embed_dim = 0
        if ftype["deep"]:
            self.deep = True
            self.deep_embedding = nn.Embedding(len(AMINO_ACIDS)+1, LEARNED_DIM, padding_idx=0).to(device)
            self.embed_dim += LEARNED_DIM
            
        if ftype["blosum"]:
            self.blosum = True
            self._build_blosum_dict()
            self.embed_dim += 20

        if ftype["onehot"]:
            self.onehot = True
            self._build_onehot_dict()
            self.embed_dim += 20

        self.a2n_func = {}
        for i, amino in enumerate(AMINO_ACIDS):
            self.a2n_func[amino] = i+1
        
        # embed peptide
        if config["embed_peptide"] == "FC":
            self.peptide_model = nn.Sequential(
                                  nn.Flatten(),
                                  nn.Linear(self.embed_dim * PEP_LENGTH, self.hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(self.hidden_dim, self.hidden_dim)).to(device)

        elif config["embed_peptide"] == "CNN":
            self.peptide_model = nn.Sequential(
                                  nn.Conv1d(self.embed_dim, self.hidden_dim, self.kmer),
                                  nn.ReLU(),
                                  nn.Flatten(),
                                  nn.Linear(self.hidden_dim * (PEP_LENGTH - self.kmer + 1), self.latent_dim)).to(device)
        elif config["embed_peptide"] == "LSTM":
            self.peptide_model = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True).to(device)
        
        # embed peptides
        self.tcr_model = nn.LSTM(self.embed_dim, self.hidden_dim, batch_first=True, bidirectional=True).to(device)

    def _build_blosum_dict(self):
        dict_file = open(BLOSUM, 'r')
        
        blosum_dict = np.zeros((len(AMINO_ACIDS)+1, len(AMINO_ACIDS)))
        for i, line in enumerate(dict_file.readlines()):
            if i == 0 or i == 21: continue
            elem = line.strip().split("\t")
            blosum_dict[i, :] = np.array(elem[1:-1]).astype(float)
            
        self.blosum_dict = torch.tensor(blosum_dict).to(device)

    def _build_onehot_dict(self):
        onehot_dict = np.zeros((len(AMINO_ACIDS)+1, len(AMINO_ACIDS)))
        
        for idx, amino in enumerate(AMINO_ACIDS):
            embed = np.zeros((len(AMINO_ACIDS)))
            embed[idx] = 1
            onehot_dict[idx+1,:] = embed

        self.onehot_dict = torch.tensor(onehot_dict).to(device)

    def _deep_encode(self, sequences):
        deep_encodings = self.deep_embedding(sequences)
        return deep_encodings
    
    def _blosum_encode(self, sequences):
        h = sequences.shape[0]
        w = sequences.shape[1]
        k = self.blosum_dict.shape[1]
        
        if len(sequences[0]) == 1: sequences = [sequences]
        
        blosum_encodings = []
        for sequence in sequences:
            blosum_encodings.append(self.blosum_dict[sequence.view(-1), :])
        
        blosum_encodings = torch.stack(blosum_encodings, axis=0)
        return blosum_encodings.float()
    
    def _onehot_encode(self, sequences):
        h = sequences.shape[0]
        w = sequences.shape[1]
        k = self.onehot_dict.shape[1]
        
        if len(sequences[0]) == 1: sequences = [sequences]

        onehot_encodings = []
        for sequence in sequences:
            onehot_encodings.append(self.onehot_dict[sequence.view(-1), :])

        onehot_encodings = torch.stack(onehot_encodings, axis=0)
        return onehot_encodings.float()
        
    def _seq2num(self, sequences, is_peptide=False):
        """ Convert sequences to interger values 
        example: "ACT" -> [1, 2, 3]
        """
        max_len = PEP_LENGTH if is_peptide else TCR_LENGTH
        arrays = np.zeros((len(sequences), max_len))
        lengths = np.zeros((len(sequences)))
        
        for i, seq in enumerate(sequences):
            lengths[i] = len(seq)

            for j, amino in enumerate(seq):
                arrays[i,j] = self.a2n_func[amino]

        lengths = torch.LongTensor(lengths).to(device)
        arrays = torch.LongTensor(arrays).to(device)
        return lengths, arrays
        
    def encode_sequences(self, sequences, is_peptide=True):
        """ Encode sequences into embeddings
        """
         
        seq_arrays = sequences
        #lengths, seq_arrays = self._seq2num(sequences, allele)
        
        if is_peptide:
            seq_encodings = torch.zeros((len(sequences), PEP_LENGTH, 0)).to(device)
        else:
            seq_encodings = torch.zeros((len(sequences), self.max_tcr_len, 0)).to(device)
        
        if self.deep:
            deep_encodings = self._deep_encode(seq_arrays)
            seq_encodings = torch.cat((seq_encodings, deep_encodings), axis=2)
        
        if self.blosum:
            blosum_encodings = self._blosum_encode(seq_arrays)
            seq_encodings = torch.cat((seq_encodings, blosum_encodings), axis=2)

        if self.onehot:
            onehot_encodings = self._onehot_encode(seq_arrays)
            seq_encodings = torch.cat((seq_encodings, onehot_encodings), axis=2)

        #if not allele:
        #    # add empty amino at the first position
        #    pad_encodings = torch.zeros((len(sequences), 1, seq_encodings.shape[2])).to(device)
        #    seq_encodings = torch.cat((pad_encodings, seq_encodings), axis=1)
        
        return seq_encodings.float()

    def forward(self, obs):
        if "max_tcr_len" not in self.__dict__: self.max_tcr_len = 27
        tcrs = obs[:, :self.max_tcr_len].long()
        peptides = obs[:, self.max_tcr_len:].long()
        
        peptides_embeddings = self.embed_peptides(peptides, is_peptide=True)
        tcrs_embeddings = self.embed_peptides(tcrs, is_peptide=False)
        
        if self.use_step:
            times = obs[:,-1].float()
            return tcrs_embeddings, peptides_embeddings, times
        else:
            return tcrs_embeddings, peptides_embeddings
            
    def embed_alleles(self, alleles):
        """ Embed allele sequences
        """
        allele_encodings = self.encode_sequences(alleles, allele=True)
        
        allele_embeddings = self.allele_model(allele_encodings.transpose(1, 2))
        return allele_embeddings

    def _get_lengths(self, sequences):
        lengths = torch.sum((sequences != 0), axis=1)
        
        return lengths
        
    def embed_peptides(self, peptides, is_peptide=False):
        peptide_encodings = self.encode_sequences(peptides, is_peptide=is_peptide)
        
        lengths = self._get_lengths(peptides)
        
        packed_encodings = torch.nn.utils.rnn.pack_padded_sequence(peptide_encodings, batch_first=True, lengths=lengths.cpu(), enforce_sorted=False)
        packed_encodings.to(device)
        
        if is_peptide:
            h0 = torch.randn(1, len(peptides), self.hidden_dim).repeat(2,1,1).to(device)
            c0 = torch.randn(1, len(peptides), self.hidden_dim).repeat(2,1,1).to(device)
            
            _, (hn, _) = self.peptide_model(packed_encodings, (h0, c0))
            return hn.transpose(1, 0).flatten(1)
        else:
            h0 = torch.randn(1, len(peptides), self.hidden_dim).repeat(2,1,1).to(device)
            c0 = torch.randn(1, len(peptides), self.hidden_dim).repeat(2,1,1).to(device)
            
            peptide_embeddings, (hn, _) = self.tcr_model(packed_encodings, (h0, c0))

            peptide_embeddings = torch.nn.utils.rnn.pad_packed_sequence(peptide_embeddings)
              
            return peptide_embeddings, hn


if __name__ == "__main__":
    # Test SeqEmbed Class
    
    alleles = ["YYAEYRNIYDTIFVDTLYIAYWFYTWAAWNYEWY", "YSEMYRERAGNTFVNTLYIWYRDYTWAVFNYLGY"]
    peptides = ["KKKHGMGKVGK", "KKADPAYGK"]

    ftype = {"deep":True, "blosum":True, "onehot": True}
    config = {"ftype":ftype, "embed_dim":60, \
              "hidden_dim":10, "latent_dim":10, \
              "kmer":3, "embed_allele":'CNN'}
    seq_embed = SeqEmbed(config)
    
    allele_embedding = seq_embed.embed_alleles(alleles)
    peptide_embedding = seq_embed.embed_peptides(peptides)
