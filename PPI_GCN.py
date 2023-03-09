#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import dgl
import torch
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from Bio.PDB import *
from sklearn.metrics import f1_score, precision_score, recall_score, matthews_corrcoef

from graphein.construct_graphs import ProteinGraph



with open('GCN/all_dset_list.pkl', 'rb') as f:
    index = pickle.load(f)

with open('GCN/dset186_label.pkl', 'rb') as f:
    dset186_labels = pickle.load(f)

with open('GCN/dset164_label.pkl', 'rb') as f:
    dset164_labels = pickle.load(f)

with open('GCN/dset72_label.pkl', 'rb') as f:
    dset72_labels = pickle.load(f)

labels = dset186_labels + dset164_labels + dset72_labels



with open('GCN/dset186_pssm_data.pkl', 'rb') as f:
    dset_186_pssms = pickle.load(f)

with open('GCN/dset164_pssm_data.pkl', 'rb') as f:
    dset_164_pssms = pickle.load(f)

with open('GCN/dset72_pssm_data.pkl', 'rb') as f:
    dset_72_pssms = pickle.load(f)

pssms = dset_186_pssms + dset_164_pssms + dset_72_pssms


df = pd.DataFrame(index)
df.columns = ['pos_index', 'example_index', 'res_position', 'dataset', 'pdb', 'length']

df = df.loc[df['res_position'] == 0]

df[['pdb_code', 'chains']] = df.pdb.str.split("_", expand=True)
df['pdb_code'] = df['pdb_code'].str.lower()
 
    
df.loc[df['dataset'] == 'dset164', 'pdb_code'] = df.copy().loc[df['dataset'] == 'dset164']['pdb'].str.slice(stop=4)
df.loc[df['dataset'] == 'dset164', 'chains'] = df.copy().loc[df['dataset'] == 'dset164']['pdb'].str.slice(-1)
df['chains'] = df['chains'].fillna('all')
obsolete = ['3NW0', '3VDO']
replacements = ['', '4NQW']
df = df.loc[~df['pdb_code'].isin(obsolete)]

with open('GCN/training_list.pkl', 'rb') as f:
    train = pickle.load(f)

with open('GCN/testing_list.pkl', 'rb') as f:
    test = pickle.load(f)

df.loc[df['pos_index'].isin(train), 'train'] = 1
df.loc[df['pos_index'].isin(test), 'train'] = 0
df.reset_index(inplace=True)

df['pdb_code'] = df['pdb_code'].str.lower()
df


pg = ProteinGraph(granularity='CA',
                  insertions=False,
                  keep_hets=False,
                  node_featuriser='meiler',
                  get_contacts_path='/home/haneen/getcontacts',
                  pdb_dir='ppisp_pdbs/',
                  contacts_dir='ppisp_contacts/',
                  exclude_waters=True,
                  covalent_bonds=False,
                  include_ss=True,
                  include_ligand=False,
                  edge_distance_cutoff=None)
graph_list = []
label_list = []
test_indices = []
train_indices = []
idx_counter = 0
for example in tqdm(range(len(labels))):
    try:
        g = pg.dgl_graph_from_pdb_code(pdb_code=df['pdb_code'][example],
                                       chain_selection=list(df['chains'][example]),
                                       edge_construction=['contacts']
                                       )
        df_index = df.iloc[example]['example_index']
        label = labels[df_index]
        pssm = pssms[df_index]
                
    except:
        print(f'Failed on example {example}')
        break
    
    if g.number_of_nodes() != len(label):
        print('label length does not match ', example)
        print(g.number_of_nodes())
        print(len(label))
        continue
    if g.number_of_nodes() != len(pssm):
        print(g.number_of_nodes())
        print(len(pssm))
        print('pssm length does not match', example)
        continue

    if df['train'][example] == 0:
        test_indices.append(idx_counter)
    if df['train'][example] == 1:
        train_indices.append(idx_counter)
    idx_counter += 1
    
    g.ndata['feats'] = torch.cat((g.ndata['h'],
                               g.ndata['ss'],
                               g.ndata['asa'],
                               g.ndata['rsa'],
                               g.ndata['coords'],
                               torch.Tensor(pssm)), dim=1)
    graph_list.append(g)
    
    label = torch.Tensor(label).long()
    label_list.append(label)


test_graphs = [graph_list[i] for i in test_indices]
train_graphs = [graph_list[i] for i in train_indices]

print(f"Train graphs: {len(train_graphs)}")
print(f"Test graphs: {len(test_graphs)}")

train_labels = [label_list[i] for i in train_indices]
test_labels = [label_list[i] for i in test_indices]

train_feats = torch.cat([graph.ndata['feats'] for graph in train_graphs], dim=0)
max_feats = torch.max(train_feats, dim=0)[0]
min_feats = torch.min(train_feats, dim=0)[0]

max_feats[max_feats == 0] = 1
for g in train_graphs:
    g.ndata['feats'] -= min_feats
    g.ndata['feats'] /= max_feats
    
for g in test_graphs:
    g.ndata['feats'] -= min_feats
    g.ndata['feats'] /= max_feats




def collate(samples):
  
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs, node_attrs='feats')
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.cat(labels)

train_data = list(zip(train_graphs, train_labels))
test_data = list(zip(test_graphs, test_labels))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True,
                         collate_fn=collate)



import torch.nn as nn
import torch.nn.functional as F

def gcn_message(edges):
  
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
   
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
       
        g.ndata['h'] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop('h')
        return self.linear(h)


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h

net = GCN(41, 16, 2)


optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
train_logits = []

epochs = 1000
epoch_losses = []

epoch_f1_scores = [] 
epoch_precision_scores = []
epoch_recall_scores = []

loss_fn = nn.NLLLoss(weight=torch.Tensor([1, 5.84]))

net.train()
for epoch in range(epochs):
    epoch_loss = 0
    
    epoch_logits = []
    labs = []
    
    for i, (bg, labels) in enumerate(train_loader):
        logits = net(bg, bg.ndata['feats'])
        train_logits.append(logits.detach().numpy())
        epoch_logits.append(logits.detach().numpy())
        labs.append(labels.unsqueeze(1).detach().numpy())

        logp = F.log_softmax(logits, 1)
        loss = loss_fn(logp, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        
    epoch_logits = np.vstack(epoch_logits)
    labs = np.vstack(labs)
    
    #print(np.argmax(epoch_logits, axis=1).sum())
    
    f1 = f1_score(labs, np.argmax(np.vstack(epoch_logits), axis=1), average='weighted')
    precision = precision_score(labs, np.argmax(np.vstack(epoch_logits), axis=1), average='weighted')
    recall = recall_score(labs, np.argmax(np.vstack(epoch_logits), axis=1), average='weighted')
    
    
    epoch_loss /= (i+1)
    if epoch % 5 == 0:
        print('Epoch %d | Loss: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f' % (epoch, epoch_loss, f1, precision, recall))
        
    epoch_losses.append(epoch_loss)
    epoch_f1_scores.append(f1)
    epoch_precision_scores.append(precision)
    epoch_recall_scores.append(recall)



test_loss =0
test_logits = []
preds = []
labs = []

net.eval()
for i, (bg, labels) in enumerate(test_loader):
    logits = net(bg, bg.ndata['feats'])
    
    test_logits.append(logits.detach().numpy())
    labs.append(labels.unsqueeze(1).detach().numpy())

    
test_logits = np.vstack(test_logits)
labs = np.vstack(labs)

f1 = f1_score(labs, np.argmax(np.vstack(test_logits), axis=1), average='weighted')
precision = precision_score(labs, np.argmax(np.vstack(test_logits), axis=1), average='weighted')
recall = recall_score(labs, np.argmax(np.vstack(test_logits), axis=1), average='weighted')

print('Test: F1: %.4f | Precision: %.4f | Recall: %.4f' % (f1, precision, recall))
