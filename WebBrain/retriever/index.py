from utils import get_argument_parser, Text_Dataset, IndexDictOfArray, show_memory_info, get_dist_info
from models import ReGenLightning
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from torch import distributed as dist
from typing import Union
import torch
import pickle
import os 
import shutil

def get_corpus(args):    
    texts = []
    idx_map = []
    for i,line in enumerate(tqdm(open(args.corpus_path))):
        line = json.loads(line)
        texts.append([i, line["contents"]])
        idx_map.append(line["id"])
    datasets = Text_Dataset(args, loaded_data=texts)
    return datasets, idx_map

def get_index(model, args):
    if os.path.exists(args.splade_tmp):
        shutil.rmtree(args.splade_tmp)
    os.makedirs(args.splade_tmp)

    manager = pl.Trainer(accelerator='auto', strategy="ddp", precision=16)
    count = 0
    idx_map = []
    dataset, idx_map = get_corpus(args)
    data_loader = DataLoader(
        dataset,
        batch_size=args.index_batch_size,
        num_workers=4
    )
    res = manager.predict(model, data_loader)
    show_memory_info("current memory:")
    rank, ws = get_dist_info()
    with open(os.path.join(args.splade_tmp, f"emb_{rank}.pkl"), 'wb') as f:
        pickle.dump(res, f)
    pickle.dump(idx_map, open(os.path.join(args.splade_tmp, "doc_ids.pkl"), "wb"))

def get_indexer(args):
    sparse_index = IndexDictOfArray(args.splade_index,
                                dim_voc=len(args.tokenizer), 
                                force_new=True)
    idx_map = pickle.load(open(os.path.join(args.splade_tmp, "doc_ids.pkl"), "rb"))
    count = 0
    doc_ids = []
    for root, dirs, files in os.walk(args.splade_tmp):
        for f in files:
            if f == "doc_ids.pkl": continue
            pickle_path = os.path.join(root, f)
            data = pickle.load(open(pickle_path,"rb"))

            for batch_res in tqdm(data):
                row = batch_res[0]
                col = batch_res[1]
                value = batch_res[2]
                idx = batch_res[3]
                row = row + count
                count += len(idx)
                sparse_index.add_batch_document(
                    row.cpu().numpy(),
                    col.cpu().numpy(),
                    value.cpu().numpy(),
                    n_docs=len(idx)
                )
                doc_ids.extend([idx_map[i] for i in idx])
            show_memory_info("current memory:")
        
        print("index contains {} documents".format(len(doc_ids)))
        
    sparse_index.save()
    pickle.dump(doc_ids, open(os.path.join(args.splade_index, "doc_ids.pkl"), "wb"))
    print("Done iterating over the corpus!")
    print("index contains {} posting lists".format(len(sparse_index)))
    print("index contains {} documents".format(len(doc_ids)))


if __name__ == "__main__":
    args = get_argument_parser()
    model = ReGenLightning(args.splade_doc_plm)
    model.freeze()
    get_index(model, args)
    dist.barrier()
    rank, ws = get_dist_info()
    if rank == 0:
        get_indexer(args)
    