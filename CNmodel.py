import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import DataLoader
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
import math
import csv
import random
import heapq
import re
import matplotlib.pyplot as plt
from Bio import SeqIO
from collections import Counter
import Biodata

class model(nn.Module):
    def __init__(self, label_num, other_feature_dim, K=2, d=1, node_hidden_dim=3, gcn_dim=128, gcn_layer_num=4, cnn_dim=64, cnn_layer_num=3, cnn_kernel_size=8, fc_dim=100, dropout_rate=0.2, pnode_nn=True, fnode_nn=True):
        super(model, self).__init__()
        self.label_num = label_num
        self.pnode_dim = d
        self.pnode_num = 4 ** (2 * K)
        self.fnode_num = 4 ** K
        self.node_hidden_dim = node_hidden_dim
        self.gcn_dim = gcn_dim
        self.gcn_layer_num = gcn_layer_num
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim
        self.dropout = dropout_rate
        self.pnode_nn = pnode_nn
        self.fnode_nn = fnode_nn
        self.other_feature_dim = other_feature_dim

        self.pnode_d = nn.Linear(self.pnode_num * self.pnode_dim, self.pnode_num * self.node_hidden_dim)
        self.fnode_d = nn.Linear(self.fnode_num, self.fnode_num * self.node_hidden_dim)
        
        self.gconvs_1 = nn.ModuleList()
        self.gconvs_2 = nn.ModuleList()
        
        if self.pnode_nn:
            pnode_dim_temp = self.node_hidden_dim
        else:
            pnode_dim_temp = self.pnode_dim
        
        if self.fnode_nn:
            fnode_dim_temp = self.node_hidden_dim
        else:
            fnode_dim_temp = 1
        
        for l in range(self.gcn_layer_num):
            if l == 0:
                self.gconvs_1.append(pyg_nn.SAGEConv((fnode_dim_temp, pnode_dim_temp), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, fnode_dim_temp), self.gcn_dim))
            else:                                   
                self.gconvs_1.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
                self.gconvs_2.append(pyg_nn.SAGEConv((self.gcn_dim, self.gcn_dim), self.gcn_dim))
        
        self.lns = nn.ModuleList()
        for l in range(self.gcn_layer_num-1):
            self.lns.append(nn.LayerNorm(self.gcn_dim))

        self.convs = nn.ModuleList()
        for l in range(self.cnn_layer_num):
            if l == 0:
                self.convs.append(nn.Conv1d(in_channels=self.gcn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
            else:
                self.convs.append(nn.Conv1d(in_channels=self.cnn_dim, out_channels=self.cnn_dim, kernel_size=self.cnn_kernel_size))
        
        if self.other_feature_dim:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim + self.other_feature_dim, self.fc_dim + self.other_feature_dim)
            self.d3 = nn.Linear(self.fc_dim + self.other_feature_dim, self.label_num)
        else:
            self.d1 = nn.Linear((self.pnode_num - (self.cnn_kernel_size - 1) * self.cnn_layer_num) * self.cnn_dim, self.fc_dim)
            self.d2 = nn.Linear(self.fc_dim, self.label_num)


    def forward(self, data):
        x_f = data.x_src
        x_p = data.x_dst
        edge_index_forward = data.edge_index[:,::2]
        edge_index_backward = data.edge_index[[1, 0], :][:,1::2]

        if self.other_feature_dim:
            other_feature = torch.reshape(data.other_feature, (-1, self.other_feature_dim)) 
        
        # transfer primary nodes
        if self.pnode_nn:
            x_p = torch.reshape(x_p, (-1, self.pnode_num * self.pnode_dim))
            x_p = self.pnode_d(x_p)
            x_p = torch.reshape(x_p, (-1, self.node_hidden_dim))
        else:
            x_p = torch.reshape(x_p, (-1, self.pnode_dim))
        
        # transfer feature nodes
        if self.fnode_nn:
            x_f = torch.reshape(x_f, (-1, self.fnode_num))
            x_f = self.fnode_d(x_f)
            x_f = torch.reshape(x_f, (-1, self.node_hidden_dim))
        else:
            x_f = torch.reshape(x_f, (-1, 1))

        for i in range(self.gcn_layer_num):
            x_p = self.gconvs_1[i]((x_f, x_p), edge_index_forward)
            x_p = F.relu(x_p)
            x_p = F.dropout(x_p, p=self.dropout, training=self.training)
            x_f = self.gconvs_2[i]((x_p, x_f), edge_index_backward)
            x_f = F.relu(x_f)
            x_f = F.dropout(x_f, p=self.dropout, training=self.training)
            if not i == self.gcn_layer_num - 1:
                x_p = self.lns[i](x_p)
                x_f = self.lns[i](x_f)

        x = torch.reshape(x_p, (-1, self.gcn_dim, self.pnode_num))
        
        for i in range(self.cnn_layer_num):
            x = self.convs[i](x)
            x = F.relu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.other_feature_dim:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(torch.cat([x, other_feature], 1))
            x = F.relu(x)
            x = self.d3(x)
            out = F.softmax(x, dim=1)

        else:
            x = x.flatten(start_dim = 1)
            x = self.d1(x)
            x = F.relu(x)
            x = self.d2(x)
            out = F.softmax(x, dim=1)

        return out


def train(dataset, model, learning_rate=1e-4, batch_size=64, epoch_n=100, random_seed=111, val_split=0.3, weighted_sampling=True, model_name="h-12.pt", device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    random.seed(random_seed)
    data_list = list(range(0, len(dataset)))
    test_list = random.sample(data_list, int(len(dataset) * val_split))
    trainset = [dataset[i] for i in data_list if i not in test_list]
    testset = [dataset[i] for i in data_list if i in test_list]
    
    if weighted_sampling:
        label_count = Counter([int(data.y) for data in dataset])
        weights = [100/label_count[int(data.y)] for data in trainset]
        sampler = WeightedRandomSampler(weights, num_samples=len(trainset), replacement=True)
        train_loader = DataLoader(trainset, batch_size=batch_size,follow_batch=['x_src', 'x_dst'], sampler=sampler)
        print("trainset length: ", len(train_loader.dataset))
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False, follow_batch=['x_src', 'x_dst'])
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, follow_batch=['x_src', 'x_dst'])
    print("testset length: ", len(test_loader.dataset))


    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_accuracies = []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0
    for epoch in range(epoch_n):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            label = batch.y

            # Forward pass
            pred = model(batch)
            loss = criterion(pred, label)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (torch.argmax(pred, 1) == label).float().mean().item()

        # Calculate epoch metrics
        avg_epoch_loss = epoch_loss / len(train_loader)
        avg_epoch_acc = epoch_acc / len(train_loader)
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_acc)

        # Validation
        val_acc = evaluation(test_loader, model, device)
        val_accuracies.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            torch.save(model, model_name)
            best_val_acc = val_acc

        print(f"Epoch {epoch+1}/{epoch_n} | Loss: {avg_epoch_loss:.4f} | "
              f"Train Acc: {avg_epoch_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()
    
    plt.tight_layout()
    # plt.savefig('learning_curves.png')
    plt.show()

    return model


def evaluation(loader, model, device):
    model.eval()
    correct = 0
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred = model(data)
            pred = pred.argmax(dim=1)
            label = data.y
        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)
    acc = correct / total

    return acc


# def test(data1, model_name="h-12.pt",val_split=1, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
#     data_list = list(range(0, len(data1)))
#     test_list = random.sample(data_list, int(len(data1) * val_split))
#     testset = [data1[i] for i in data_list if i in test_list]
#     model = torch.load(model_name, map_location=device)
#     loader = DataLoader(testset, batch_size=len(data1), shuffle=False, follow_batch=['x_src', 'x_dst'])

#     model.eval()
#     TP, FN, FP, TN = 0, 0, 0, 0

#     for data in loader:
#         with torch.no_grad():
#             data = data.to(device)
#             pred = model(data)
#             pred = pred.argmax(dim=1)
#             label = data.y
#             AUC = Calauc(label, pred)
#             # correct += pred.eq(label).sum().item()
#             A, B, C, D = eff(label, pred)
#             TP += A
#             FN += B
#             FP += C
#             TN += D


#     SN, SP, ACC = Judeff(TP, FN, FP, TN)
    
#     # print("TP: {}, FN: {}, FP: {}, TN: {}".format(TP, FN, FP, TN))
#     # print("SN: {}, SP: {}, ACC: {}, AUC: {}".format(SN, SP, ACC, AUC))

#     return {
#         "TP": TP, "FN": FN, "FP": FP, "TN": TN,
#         "SN": SN, "SP": SP, "ACC": ACC, "AUC": AUC
#     }


def test(data1, model_name="h-12.pt", val_split=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), fasta_file=None):
    data_list = list(range(0, len(data1)))
    test_list = random.sample(data_list, int(len(data1) * val_split))
    testset = [data1[i] for i in data_list if i in test_list]
    model = torch.load(model_name, map_location=device)
    loader = DataLoader(testset, batch_size=len(data1), shuffle=False, follow_batch=['x_src', 'x_dst'])

    # Read gene IDs from FASTA file if provided
    gene_ids = []
    if fasta_file:
        gene_ids = [seq_record.id for seq_record in SeqIO.parse(fasta_file, "fasta")]
        # Ensure we only take the genes in our test set
        gene_ids = [gene_ids[i] for i in data_list if i in test_list]

    model.eval()
    TP, FN, FP, TN = 0, 0, 0, 0
    all_probs = []  # Stores probability of being essential (class 1)
    all_labels = []  # Actual labels
    all_preds = []   # Predicted labels (0 or 1)

    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred_probs = model(data)
            preds = pred_probs.argmax(dim=1)
            labels = data.y
            
            all_probs.extend(pred_probs[:, 1].cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            AUC = Calauc(labels, preds)
            A, B, C, D = eff(labels, preds)
            TP += A
            FN += B
            FP += C
            TN += D

    SN, SP, ACC = Judeff(TP, FN, FP, TN)
    
    results = {
        "metrics": {
            "TP": TP, "FN": FN, "FP": FP, "TN": TN,
            "SN": SN, "SP": SP, "ACC": ACC, "AUC": AUC
        },
        "predictions": {
            "probabilities": all_probs,
            "predicted_labels": all_preds,
            "true_labels": all_labels
        }
    }
    
    if gene_ids:
        results["predictions"]["gene_ids"] = gene_ids
    
    return results


def predict(fasta_file, model_name, K=2, d=1, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """Predict essentiality for genes in a new FASTA file"""
    # 1. Load model
    model = torch.load(model_name, map_location=device)
    model.eval()
    
    # 2. Prepare data (without labels)
    bio_data = Biodata.Biodata(
        fasta_file=fasta_file,
        label_file=None,  # No labels available
        feature_file=None,
        K=K,
        d=d
    )
    dataset = bio_data.encode(thread=48)
    
    # 3. Get predictions
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, follow_batch=['x_src', 'x_dst'])
    
    gene_ids = [seq_record.id for seq_record in SeqIO.parse(fasta_file, "fasta")]
    all_probs = []
    
    for data in loader:
        with torch.no_grad():
            data = data.to(device)
            pred_probs = model(data)
            all_probs.extend(pred_probs[:, 1].cpu().numpy())  # Probability of being essential
    
    # 4. Create and sort results
    results = pd.DataFrame({
        'Gene_ID': gene_ids,
        'Probability_Essential': all_probs
    }).sort_values('Probability_Essential', ascending=False).reset_index(drop=True)
    
    return results


def eff(labels, preds):

    TP, FN, FP, TN = 0, 0, 0, 0
    for idx,label in enumerate(labels):

        if label == 1:
            if label == preds[idx]:
                TP += 1
            else: FN += 1
        elif label == preds[idx]:
            TN += 1
        else: FP += 1

    return TP, FN, FP, TN


def Judeff(TP, FN, FP, TN):

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + FN + FP + TN)

    return SN, SP, ACC

def Calauc(labels, preds):

    labels = labels.clone().detach().cpu().numpy()
    preds = preds.clone().detach().cpu().numpy()

    f = list(zip(preds, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    pos_cnt = np.sum(labels == 1)
    neg_cnt = np.sum(labels == 0)
    AUC = (np.sum(rankList) - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)

    return AUC