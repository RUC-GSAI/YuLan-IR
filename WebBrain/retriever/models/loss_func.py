import torch
import torch.nn as nn

def ranking_loss(query_embs, pos_doc_embs, neg_doc_embs):
    batch_size = len(query_embs)
    pos_scores = query_embs.mm(pos_doc_embs.T)  # B * B
    score_mat = pos_scores
    if neg_doc_embs is not None:
        neg_scores = torch.sum(query_embs.unsqueeze(1) * neg_doc_embs, dim = -1) # B * neg_ratio
        score_mat = torch.cat([pos_scores, neg_scores], dim = 1)    # B * (B + neg_ratio)  in_batch negatives + neg_ratio other negatives
    label_mat = torch.arange(batch_size).to(query_embs.device)
    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(score_mat, label_mat)
    return loss
