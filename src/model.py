from math import floor, ceil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from local_config import *


torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


class CRCNNModel(nn.Module):
    def __init__(self, word_vectors, rel2id):
        super(CRCNNModel, self).__init__()
        self.rel2id = rel2id

        self.word_embedding = nn.Embedding.from_pretrained(word_vectors, freeze=True)
        self.pos_embedding = nn.Embedding(MAX_SENT_LEN*2, POS_EMBEDDING_DIM)
        pads = (KERNEL_SIZE-1)/2
        if pads - floor(pads) > 0.0:
            padding = (floor(pads), ceil(pads))
        else:
            padding = int(pads)
        self.conv = nn.Conv1d(WORD_EMBEDDING_DIM + 2*POS_EMBEDDING_DIM, FILTER_NUM, KERNEL_SIZE, padding=padding)
        self.max_pool = nn.MaxPool1d(MAX_SENT_LEN, 1)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
    
    def forward(self, tokens, pos1, pos2):
        word_embedding_layer = self.word_embedding(tokens)
        pos1_embedding_layer = self.pos_embedding(pos1)
        pos2_embedding_layer = self.pos_embedding(pos2)
        concat_layer = torch.cat([word_embedding_layer, pos1_embedding_layer, pos2_embedding_layer], dim=-1)
        out = concat_layer.permute(0, 2, 1)
        out = self.conv(out)
        out = F.tanh(out)
        out = self.max_pool(out)
        out = out.view(-1, FILTER_NUM)

        scope = np.sqrt(6/(len(self.rel2id) + FILTER_NUM))
        W_classes = torch.tensor(np.random.uniform(-scope, scope, (FILTER_NUM, len(self.rel2id))), 
                                    requires_grad=True, dtype=out.dtype, device=DEVICE)
        out = out.mm(W_classes) # dimension: BATCH_SIZE*len(self.rel2id)
        return out


class RankingLoss(nn.Module):
    def __init__(self, rel2id,):
        super(RankingLoss, self).__init__()
        self.rel2id = rel2id
        self.margin_positive = MARGIN_POSITIVE
        self.margin_negative = MARGIN_NEGATIVE
        self.gamma = GAMMA_SCALING_FACTOR

    def forward(self, scores, ground_true_labels):
        ground_true_labels = ground_true_labels.view(BATCH_SIZE, -1)
        scores_positive = scores[:, ground_true_labels.tolist()[0]]
        loss_positive = torch.log(1.0 + torch.exp(self.gamma * ( self.margin_positive - scores_positive )))

        # scores_negative: the max score from all the negative labels
        sp = [x[0] for x in scores_positive.tolist()]
        ss = scores.tolist()
        for i in range(len(ss)):
            ss[i].remove(sp[i])
        sn = torch.tensor(ss, dtype=scores_positive.dtype, requires_grad=True, device=DEVICE)
        scores_negative, _ = torch.max(sn, dim=-1)
        scores_negative[(ground_true_labels == self.rel2id['Other']).tolist()[0]] = 0.0
        loss_negative = torch.log(1.0 + torch.exp(self.gamma * ( self.margin_negative + scores_negative )))
        loss = torch.mean(loss_positive + loss_negative)
        return loss
