from torchmetrics import RetrievalRecall, RetrievalMRR, RetrievalPrecision, RetrievalMAP
from torch import tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import Text_Dataset, get_argument_parser, \
                    SparseRetrieval
import json
import faiss
import pytorch_lightning as pl
from transformers import AutoTokenizer
from models import ReGenLightning
import torch
import os
import numpy as np
import math

def compute_metrics(indexes, preds, target):
    r1 = RetrievalRecall(k=1)
    p1 = RetrievalPrecision(k=1)
    p5 = RetrievalPrecision(k=5)
    r5 = RetrievalRecall(k=5)
    r10 = RetrievalRecall(k=10)
    r20 = RetrievalRecall(k=20)
    mrr = RetrievalMRR()
    rmap = RetrievalMAP()
    rtn = {}
    rtn["recall@1"] =  r1(preds, target, indexes=indexes).item()
    rtn["precision@1"] = p1(preds, target, indexes=indexes).item()
    rtn["recall@5"] = r5(preds, target, indexes=indexes).item()
    rtn["precision@5"] = p5(preds, target, indexes=indexes).item()
    rtn["recall@10"] = r10(preds, target, indexes=indexes).item()
    rtn["recall@20"] = r20(preds, target, indexes=indexes).item()
    rtn["mrr"] = mrr(preds, target, indexes=indexes).item()
    rtn["map"] = rmap(preds, target, indexes=indexes).item()
    return rtn

def get_test_data(test_path):
    from collections import defaultdict
    qrels = defaultdict(list)
    for i,line in tqdm(enumerate(open(test_path))):
        # if i > 500:break
        line = line.strip().split("\t")
        qrels[line[1]].append(line[2])
    return qrels

def get_query_dl(test_path, args):
    test_data = get_test_data(test_path)
    queries = []
    for i,k in enumerate(test_data):
        queries.append([i,k])

    query_data = Text_Dataset(args, loaded_data=queries, is_query=True)
    query_loader = DataLoader(
            query_data,
            batch_size=args.encode_batch_size,
            num_workers=4
        )
    return query_loader, test_data


def evaluate_splade(test_path, trainer, model, args):
    query_loader, test_data = get_query_dl(test_path, args)
    res = trainer.predict(model, query_loader)
    embs = [b[0] for b in res]
    embs = torch.cat(embs, dim=0).cpu()
    qid2emb = {}
    for i,emb in enumerate(embs):
        qid2emb[i] = emb
    retriever = SparseRetrieval(
        args.splade_index,
        dim_voc=args.tokenizer.vocab_size,
        top_k=20,
        retrieval_output_path=args.splade_output_path)

    res = retriever.retrieve(qid2emb)
    pids = list(test_data.values())

    indexes = []
    preds = []
    target = []
    test_data

    for qid in res:
        postive_ids = pids[int(qid)]
        for pid in res[qid]:
            indexes.append(int(qid))
            preds.append(res[qid][pid])
            if pid in postive_ids:
                target.append(True)
            else:
                target.append(False)
    return compute_metrics(
        tensor(indexes), 
        tensor(preds), 
        tensor(target))


if __name__ == "__main__":
    args = get_argument_parser()
    net = ReGenLightning(args.splade_query_plm, cp=args.checkpoint, out_type="dense")
    net.freeze()
    trainer = pl.Trainer(accelerator='auto', strategy="ddp", precision=16)
 
    res = evaluate_splade(
        args.dev_file, 
        trainer, net, args) 

    print(res)
    with open("regen.json", 'w') as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
