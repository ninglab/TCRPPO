import argparse
import torch
import copy
import torch.nn as nn

#from config import AMINO_ACIDS
import sys, os
import pdb
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import blosum_dict, seq2num, num2seq, blosum_encode

AMINO_ACIDS = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
device = "cpu" if not torch.cuda.is_available() else "cuda"


def get_likelihood(logits):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch,) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """
    if logits.shape[1] > 1:
        logits = torch.exp(logits)
        log_probs = torch.log( logits / torch.tile( logits.sum(1).unsqueeze(1), (1, logits.size(1))) + 1e-10)
    else:
        logits = torch.exp(-logits)
        log_probs_1 = torch.log( 1 / (1 + logits) )
        log_probs_0 = torch.log( 1 - 1 / (1 + logits) + 1e-10 )
        
        #pdb.set_trace()
        log_probs = torch.stack( [ log_probs_0, log_probs_1 ], dim=1)
        #if torch.isinf(log_probs).any(): pdb.set_trace()
    
    return log_probs


class TCRLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, latent_size, encode_method, beta=0.1, learned_size=20, min_len=8, max_len=35, bidirectional=True):
        super().__init__()

        self.min_len = min_len
        self.max_len = max_len

        self.encode_method = encode_method
        
        self.bidirectional = bidirectional
        
        self.embedding_size = embed_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = len(AMINO_ACIDS) + 1

        self.embedding = nn.Embedding(len(AMINO_ACIDS)+1, embed_size, padding_idx=0).to(device)

        # LSTM weights
        self.decoder_i = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_o = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_f = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Sigmoid() ).to(device)
        self.decoder_g = nn.Sequential( nn.Linear( embed_size + hidden_size, hidden_size), nn.Tanh() ).to(device)
        
        
        self.hidden2out = nn.Sequential(
                               nn.Linear(embed_size + hidden_size, hidden_size),
                               nn.ReLU(),
                               nn.Linear(hidden_size, self.output_size)).to(device)

        self.loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, sequences):
        hidden = torch.zeros((len(sequences), self.hidden_size)).to(device)
        cell = torch.zeros_like(hidden).to(device)

        init_x_vecs = torch.zeros((len(sequences), self.embedding_size)).to(device)
        
        amino_hiddens, targets = [], []
        
        #for i, seq in enumerate(input_sequences):
        #    input_sequences[i] = seq[::-1]
        
        seq_arrays, length = seq2num(sequences)
        #encoded_arrays = blosum_encode(seq_arrays)
        
        for t in range(self.max_len):
            
            tokens, nexts, nonstop_idxs = [], [], []
            for i, seq in enumerate(sequences):
                if t <= len(seq):
                    if t > 0: 
                        tokens.append( seq_arrays[i][t-1] )
                    
                    if t < len(seq):
                        if t != 0: nonstop_idxs.append(len(tokens) - 1)
                        nexts.append(seq_arrays[i][t])
                    else:
                        nexts.append(0)
            
            if t == 0: nonstop_idxs = [i for i in range(len(sequences))]
            targets.extend(nexts)
            
            if t == 0:
                x_vecs = init_x_vecs
            else:
                tokens = torch.LongTensor(tokens).to(device)
                x_vecs = self.embedding(tokens)
                
            amino_hiddens.append( torch.cat( (x_vecs, hidden.clone()), dim=-1))
            
            new_h, new_c = self.__decoder_lstm(x_vecs, hidden, cell)
            
            if len(nonstop_idxs) == 0: break
            
            nonstop_idxs = torch.LongTensor(nonstop_idxs).to(device)
            #z = torch.index_select(z, 0, nonstop_idxs)
            hidden = torch.index_select(new_h, 0, nonstop_idxs)
            cell   = torch.index_select(new_c, 0, nonstop_idxs)
        
        amino_hiddens = torch.cat(amino_hiddens, dim=0)
        amino_scores = self.hidden2out(amino_hiddens).squeeze(dim=1)
        
        targets = torch.LongTensor(targets).to(device)
        loss = self.loss(amino_scores, targets) / len(sequences)
        _, amino = torch.max(amino_scores, dim=1)
        
        amino_acc = torch.eq(amino, targets).float()
        amino_acc = torch.sum(amino_acc) / targets.shape[0]
        
        return loss, amino_acc
            
    def __decoder_lstm(self, x, h, c):
        i = self.decoder_i( torch.cat([x, h], dim=-1) )
        o = self.decoder_o( torch.cat([x, h], dim=-1) )
        f = self.decoder_f( torch.cat([x, h], dim=-1) )
        g = self.decoder_g( torch.cat([x, h], dim=-1) )
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c
        
    def test(self, sequences):
        hidden = torch.zeros((len(sequences), self.hidden_size)).to(device)
        cell = torch.zeros_like(hidden).to(device)

        init_x_vecs = torch.zeros((len(sequences), self.embedding_size)).to(device)
        
        amino_hiddens, targets = [], []
        
        #for i, seq in enumerate(input_sequences):
        #    input_sequences[i] = seq[::-1]
        
        seq_arrays, length = seq2num(sequences)
        #encoded_arrays = blosum_encode(seq_arrays)
        indices = []
        
        for t in range(self.max_len):
            
            tokens, nexts, nonstop_idxs = [], [], []
            for i, seq in enumerate(sequences):
                if t <= len(seq):
                    if t > 0: 
                        tokens.append( seq_arrays[i][t-1] )
                    
                    if t < len(seq):
                        if t != 0: nonstop_idxs.append(len(tokens) - 1)
                        nexts.append(seq_arrays[i][t])
                    else:
                        nexts.append(0)
            
            if t == 0: nonstop_idxs = [i for i in range(len(sequences))]
            targets.extend(nexts)
            
            if t == 0:
                x_vecs = init_x_vecs
            else:
                tokens = torch.LongTensor(tokens).to(device)
                x_vecs = self.embedding(tokens)
                
            amino_hiddens.append( torch.cat( (x_vecs, hidden.clone()), dim=-1))
            
            new_h, new_c = self.__decoder_lstm(x_vecs, hidden, cell)
            
            if len(nonstop_idxs) == 0: break
            
            nonstop_idxs = torch.LongTensor(nonstop_idxs).to(device)
            #z = torch.index_select(z, 0, nonstop_idxs)
            hidden = torch.index_select(new_h, 0, nonstop_idxs)
            cell   = torch.index_select(new_c, 0, nonstop_idxs)
        
        amino_hiddens = torch.cat(amino_hiddens, dim=0)
        amino_scores = self.hidden2out(amino_hiddens).squeeze(dim=1)
        likelihoods = get_likelihood(amino_scores)
        
        targets = torch.LongTensor(targets).to(device)
        
        target_likelihoods = torch.gather(likelihoods, 1, targets.unsqueeze(1))
        
        all_likelihoods = torch.zeros(len(sequences))
        num = 0
        for i, seq in enumerate(sequences):
            all_likelihoods[i] = torch.sum(target_likelihoods[num:num+len(seq)+1])
            num += (len(seq) + 1)
        
        #loss = self.loss(amino_scores, targets) / len(input_sequences)
        #_, amino = torch.max(amino_scores, dim=1)
        #
        #amino_acc = torch.eq(amino, targets).float()
        #amino_acc = torch.sum(amino_acc) / targets.shape[0]
    
        return all_likelihoods
        
    def generate(self, z):
        hidden = self.latent2hidden(z)
        cell   = torch.zeros_like(hidden).to(device)
        
        x_vecs = torch.zeros((z.shape[0], self.embedding_size)).to(device)
        padded_sequence = torch.zeros(z.shape[0], self.max_len, dtype=torch.long).to(device)
        
        idxs = torch.arange(z.shape[0]).to(device)
        
        for i in range(self.max_len):
            amino_hidden = torch.cat((x_vecs, hidden, z), dim=-1)
            
            amino_scores = self.hidden2out(amino_hidden)

            if i <= self.min_len:
                amino_scores[:, 0] = torch.ones(z.shape[0]) * -1000
            
            _, out = torch.max(amino_scores, dim=1)
            
            nonstop_idxs = torch.arange(hidden.shape[0]).to(device)
            if i > self.min_len:
                nonstop_idxs = torch.nonzero(out).squeeze(dim=1).to(device)
                
                out = torch.index_select(out, 0, nonstop_idxs)
                idxs = torch.index_select(idxs, 0, nonstop_idxs)
            
            if nonstop_idxs.shape[0] == 0: break
            padded_sequence[idxs, i] = out
               
            #.set_trace()
            out_vec = np.take(blosum_dict, out.cpu().detach().numpy().astype(int), 0)
            out_vec = torch.tensor(out_vec).float().to(device)
            
            new_hidden = torch.index_select(hidden, 0, nonstop_idxs)
            z          = torch.index_select(z, 0, nonstop_idxs)
            new_cell   = torch.index_select(cell, 0, nonstop_idxs)
            x_vecs     = torch.index_select(x_vecs, 0, nonstop_idxs)
            hidden, cell = self.__decoder_lstm(x_vecs, new_hidden, new_cell)
            x_vecs = out_vec
            
            
        seqs = num2seq(padded_sequence)
        seqs = [seq[::-1] for seq in seqs]
        return seqs

    def edit_dist(self, tcrs):
        z = self.encode(tcrs)
        seqs = self.generate(z)
        
        dists = np.zeros(len(seqs))
        for i, (seq, tcr) in enumerate(zip(seqs, tcrs)):
            dist = 0
            for j in range(min(len(seq), len(tcr))):
                if seq[j] != tcr[j]:
                    dist += 1
        
            dist += abs(len(seq) - len(tcr))
            dists[i] = dist / len(tcr)
            
        return seq, dists, z.cpu().detach().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='/users/PES0781/ziqichen/peptideproject/TCR/model/reward/model/')
    parser.add_argument("--data_path", type=str, default="/users/PES0781/ziqichen/peptideproject/TCR/tcrdb/uniq_tcr_seqs.txt")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--embed_method", type=str, default="blosum")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--latent_size", type=int, default=16)
    parser.add_argument("--test_size", type=int, default=32)
    parser.add_argument("--stop_criteria", type=int, default=20)
    parser.add_argument("--max_step", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--print_iter", type=int, default=50)
    parser.add_argument("--save_iter", type=int, default=50000)
    
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=10000)
    parser.add_argument("--rate", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0005)

    parser.add_argument("--beta_anneal_iter", type=int, default=10000)
    parser.add_argument("--max_beta", type=float, default=0.5)
    parser.add_argument("--step_beta", type=float, default=0.1)

    parser.add_argument("--clip_norm", type=float, default=50.0)
    args = parser.parse_args()

    embed_methods = ['blosum', 'onehot', 'deep']

    embedding_size = 0
    for method in embed_methods:
        if method in args.embed_method: embedding_size += 20

    model = AE(embedding_size, args.hidden_size, args.latent_size, \
                args.embed_method, beta=args.beta)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate) 
    
    num, iter_num, accs, losses = 0, 0, 0, 0
    
    train_TCRs = [line.strip() for line in open(args.data_path, 'r').readlines()]

    print("with %s device" % (device))
    print("with update rate: %.1f" % (args.rate))
    beta = args.beta
    while num < args.stop_criteria and iter_num < args.max_step:
        iter_num += 1
        
        batch_TCRs_idxs = np.random.choice(len(train_TCRs), size=args.batch_size)
        batch_TCRs = [train_TCRs[idx] for idx in batch_TCRs_idxs]
        
        with torch.autograd.set_detect_anomaly(True):
            model.zero_grad()
            
            loss, acc, _ = model(batch_TCRs)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
       
        accs = accs + float(acc)
        losses = losses + float(loss)
        
        if iter_num % args.print_iter == 0:
            test_TCR_idxs = np.random.choice(len(train_TCRs), size=args.batch_size)
            test_TCRs = [train_TCRs[idx] for idx in test_TCR_idxs]
            
            tcrs = model.generate(model.encode(test_TCRs))
            #_, _, _, new_acc, _ = model(test_peptides)
            
            test_acc = 0
            for seq1, seq2 in zip(test_TCRs, tcrs):
                if seq1 == seq2:
                    test_acc += 1
            
            test_acc /= len(tcrs)
            
            accs /= args.print_iter
            losses /= args.print_iter
            print("step: %d; accuracy: %.4f; loss: %.4f; test_acc: %.4f" % (iter_num, accs, losses, test_acc))
            sys.stdout.flush()
            accs = 0
            losses= 0
            
            #if new_acc > 0.98:
            #    num += 1
            #else:
            #    num = 0

        if iter_num % args.save_iter == 0:
            torch.save(model.state_dict(), "%s_step%d" % (args.model_path, iter_num))

        if iter_num % args.beta_anneal_iter == 0:
            beta = min(args.max_beta, beta + args.step_beta)
            model.beta = beta
            print("beta value: %.2f" % (beta))
            
        if iter_num % args.anneal_iter == 0 and scheduler.get_lr()[0] > args.min_lr:
            scheduler.step()
            print("learning rate: %.6f" % scheduler.get_lr()[0])


    torch.save(model.state_dict(), "%s.pt" % (args.model_path))
