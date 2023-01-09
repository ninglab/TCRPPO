import pdb
import numpy as np
import torch
from config import TCR_LENGTH, PEP_LENGTH, AMINO_ACIDS, BLOSUM


def onehot_encode(sequences):
    if len(sequences[0]) == 1: sequences = [sequences]
    
    onehot_mat = np.eye(20)
    onehot_encodings = []
    
    for sequence in sequences:
        onehot_encodings.append(np.take(onehot_mat, sequence.astype(int)-1, 0))
    
    onehot_encodings = np.stack(onehot_encodings, axis=0)
    return onehot_encodings


def __build_blosum_dict():
    dict_file = open(BLOSUM, 'r')

    blosum_dict = np.zeros((len(AMINO_ACIDS)+1, len(AMINO_ACIDS)))
    for i, line in enumerate(dict_file.readlines()):
        if i == 0 or i == 21: continue
        elem = line.strip().split("\t")
        blosum_dict[i, :] = np.array(elem[1:-1]).astype(float)

    return blosum_dict

a2n_func = {amino:i+1 for i, amino in enumerate(AMINO_ACIDS)}
n2a_func = {i+1:amino for i, amino in enumerate(AMINO_ACIDS)}
blosum_dict = __build_blosum_dict()

def seq2num(sequences, max_len=0):
    """ Convert sequences to interger values 
    example: "ACT" -> [1, 2, 3]
    """
    if max_len == 0: max_len = max([len(seq) for seq in sequences])
    arrays = np.zeros((len(sequences), max_len))
    lengths = np.zeros((len(sequences)))

    for i, seq in enumerate(sequences):
        lengths[i] = len(seq)

        for j, amino in enumerate(seq):
            arrays[i,j] = a2n_func[amino]
    
    return arrays, lengths

def num2seq(sequences):
    new_sequences = []
    for i in range(sequences.shape[0]):
        seq = "".join([n2a_func[idx.item()] for idx in sequences[i,:] if idx.item() != 0])
        new_sequences.append(seq)

    return new_sequences

    
def blosum_encode(sequences):
    if len(sequences[0]) == 1: sequences = [sequences]
    
    blosum_encodings = []
    
    for sequence in sequences:
        blosum_encodings.append(np.take(blosum_dict, sequence.astype(int), 0))
    
    blosum_encodings = np.stack(blosum_encodings, axis=0)
    return blosum_encodings


def edit_sequence(peptides, actions):
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
