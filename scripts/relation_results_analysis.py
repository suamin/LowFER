# -*- coding: utf-8 -*-

import torch
import collections
import numpy as np

from model import LowFER
from load_data import Data


def get_data_idxs(data):
    data_idxs = [(entity_idxs[data[i][0]], relation_idxs[data[i][1]], \
                  entity_idxs[data[i][2]]) for i in range(len(data))]
    return data_idxs


def get_er_vocab(data):
    er_vocab = collections.defaultdict(list)
    for triple in data:
        er_vocab[(triple[0], triple[1])].append(triple[2])
    return er_vocab


def get_batch(batch_size, er_vocab, er_vocab_pairs, idx, label_smoothing=0.1, cuda=True, device=torch.device('cuda')):
    batch = er_vocab_pairs[idx:idx+batch_size]
    targets = np.zeros((len(batch), len(d.entities)))
    for idx, pair in enumerate(batch):
        targets[idx, er_vocab[pair]] = 1.
    targets = torch.FloatTensor(targets)
    if label_smoothing:
        targets = ((1.0-label_smoothing)*targets) + (1.0/targets.size(1))
    if cuda:
        targets = targets.to(device)
    return np.array(batch), targets


def evaluate(model, data, batch_size=128, cuda=True, device=torch.device('cuda'), R=6):
    hits = []
    ranks = []
    for i in range(10):
        hits.append([])
    
    test_data_idxs = [(i, j, k) for i, j, k in get_data_idxs(data) if j == R]
    er_vocab = get_er_vocab(get_data_idxs(d.data))
    
    print("Number of data points: %d" % len(test_data_idxs))
    
    for i in range(0, len(test_data_idxs), batch_size):
        data_batch, _ = get_batch(batch_size, er_vocab, test_data_idxs, i)
        e1_idx = torch.tensor(data_batch[:,0])
        r_idx = torch.tensor(data_batch[:,1])
        e2_idx = torch.tensor(data_batch[:,2])
        if cuda:
            e1_idx = e1_idx.to(device)
            r_idx = r_idx.to(device)
            e2_idx = e2_idx.to(device)
        predictions = model.forward(e1_idx, r_idx)
        
        for j in range(data_batch.shape[0]):
            filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
            target_value = predictions[j,e2_idx[j]].item()
            predictions[j, filt] = 0.0
            predictions[j, e2_idx[j]] = target_value
        
        sort_values, sort_idxs = torch.sort(predictions.cpu(), dim=1, descending=True)
        
        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(data_batch.shape[0]):
            rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
            ranks.append(rank+1)
            
            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)
    
    metrics = {
        'h10': np.mean(hits[9]),
        'h3': np.mean(hits[2]),
        'h1': np.mean(hits[0]),
        'mr': np.mean(ranks),
        'mrr': np.mean(1./np.array(ranks))
    }
    print('Hits @10: {0}'.format(metrics['h10']))
    print('Hits @3: {0}'.format(metrics['h3']))
    print('Hits @1: {0}'.format(metrics['h1']))
    print('Mean rank: {0}'.format(metrics['mr']))
    print('Mean reciprocal rank: {0}'.format(metrics['mrr']))


# WN18

data_dir = '../data/WN18/'
d = Data(data_dir=data_dir, reverse=True)
entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}

de = 200
dr = 30
k = 30
# pass trained checkpoint saved during training, e.g. as
model_dir = "WN18_lr_0.005_dr_0.995_e_200_r_30_k_{}_id_0.2_hd1_0.1_hd2_0.2_ls_0.1".format(k)
model_ckpt = '../output/WN18/{}/final.pt'.format(model_dir)
model_kwargs = {'k':k, 'input_dropout':0.2, 'hidden_dropout1':0.2, 'hidden_dropout2':0.3}
model = LowFER(d, de, dr, **model_kwargs)

model.load_state_dict(torch.load(model_ckpt))
model.eval()
model = model.cuda()

for r, r_idx in relation_idxs.items():
    if r.split('_')[-1] != 'reverse':
       print(r)
       ms = evaluate(model, d.test_data, R=r_idx)
