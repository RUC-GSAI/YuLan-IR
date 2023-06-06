import re
import os
import json
import subprocess
import warnings
import pickle
import torch
import faiss
import argparse
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
from dense_model import *
from adaptertransformers.src import transformers
from adaptertransformers.src.transformers import AutoConfig, AutoTokenizer
from adaptertransformers.src.transformers import PretrainedConfig

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts[index]
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tokenized_text["input_ids"])
        attention_mask = torch.tensor(tokenized_text["attention_mask"])
        return input_ids, attention_mask

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    attention_masks = [item[1] for item in batch]
    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)
    return input_ids, attention_masks


def load_json_data(data_dir):
    """
    Load multiple JSON files from the folder and merge.
    """

    files = os.listdir(data_dir)
    files.sort()
    all_data = []
    for file in files:
        print("Loading: ",file)
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding = "utf-8") as f:
            doc = json.load(f)
        all_data.append(doc)
        #all_data += doc
    return all_data

def check_dir(dir_path):
    """
    Determine if the folder exists and if there is content.
    """
    
    if os.path.isdir(dir_path):
        if len(os.listdir(dir_path)) > 0:
            return False
    else:
        os.makedirs(dir_path)
    return True

class Index_Builder:
    """
    Build an index for retrieval based on Json data.

    :param index_type: the type of index to build can be sparse, dense, or all. 
                        If the parameter is all, two types of indexes will be built.
    :param data_dir: the folder path containing JSON files, 
                        it can contain multiple documents that need to be used.
                        The format of all documents should be consistent.
    :param index_save_dir: the folder path for saving index,
                        different types of indices will be stored in folders with their respective sub names,
                        like "dense" and "sparse".
    """

    def __init__(
            self, 
            index_type, 
            data_dir, 
            index_save_dir, 
            train_dam_flag,
            use_content_type,
            cuda_id, 
            batch_size,
            dam_path
    ):
        self.index_type = index_type   
        self.data_dir = data_dir      
        self.index_save_dir = index_save_dir   
        self.train_dam_flag = train_dam_flag 
        self.use_content_type = use_content_type
        self.cuda_id = cuda_id
        self.batch_size = batch_size
        self.dam_path = dam_path

        if not os.path.exists(index_save_dir):
            os.makedirs(index_save_dir)

    def build_index(self):
        if self.index_type == "sparse":
            self.build_sparse_index()
        elif self.index_type == "dense":
            self.build_dense_index()
        else:
            self.build_sparse_index()
            self.build_dense_index()
    
    def train_dam(self):
        """
        Training a DAM module based on unsupervised methods.
        
        Firstly, it is necessary to convert contents data in JSON files into TSV format, 
                and then use the script provided by the disentangled_retriever library for training.
        Reference: https://github.com/jingtaozhan/disentangled-retriever/blob/main/examples/domain_adapt/chinese-dureader/adapt_to_new_domain.md
        """

        # build corpus for training
        corpus_path = self.index_save_dir + '/corpus.tsv'
        with open(corpus_path,"w") as f:
            for doc in self.all_docs:
                # remove special character
                if self.use_content_type == "title":
                    doc_content = doc['title']
                elif self.use_content_type == "contents":
                    doc_content = doc['contents']
                else:
                    doc_content = doc['title'] + doc['contents']
                content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '',doc_content)
                f.write("{}\t{}\n".format(doc['id'],content))

        # train DAM module
        dam_path = self.index_save_dir + '/dam_module'
        if not check_dir(dam_path):
            warnings.warn("Overwrite trained DAM module!", UserWarning)
        training_args = ["--nproc_per_node", "1",
                            "-m", "train_dam_module",
                            "--corpus_path", corpus_path,
                            "--output_dir", dam_path,
                            "--model_name_or_path", "jingtao/DAM-bert_base-mlm-dureader",
                            "--max_seq_length", "512",
                            "--gradient_accumulation_steps", "1",
                            "--per_device_train_batch_size", "64",
                            "--warmup_steps", "1000",
                            "--fp16",
                            "--learning_rate", "2e-5",
                            "--max_steps", "20000",
                            "--dataloader_drop_last",
                            "--overwrite_output_dir",
                            "--weight_decay", "0.01",
                            "--save_steps", "5000",
                            "--lr_scheduler_type", "constant_with_warmup",
                            "--save_strategy", "steps",
                            "--optim", "adamw_torch"]
        subprocess.run(["torchrun"] + training_args)
        return dam_path

    def build_model(self, dam_path):
        """
        Building model for converting document contents into embeddings,
            the model includes DAM module and REM module.
        """

        config = PretrainedConfig.from_pretrained(dam_path)
        config.similarity_metric, config.pooling = "ip", "average"
        tokenizer = AutoTokenizer.from_pretrained(dam_path, config=config)
        model = BertDense.from_pretrained(dam_path, config=config)
        adapter_name = model.load_adapter(REM_URL)
        model.set_active_adapters(adapter_name)
        return model, tokenizer

    def build_sparse_index(self):
        """
        Building BM25 index based on Pyserini library.

        Reference: https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building-a-bm25-index-direct-java-implementation
        """

        sparse_index_path = self.index_save_dir + "/sparse"
        if not check_dir(sparse_index_path):
            warnings.warn("Sparse index already exists and will be overwritten.", UserWarning)

        print("Start building sparse index...")
        pyserini_args = ["--collection", "JsonCollection",
                         "--input", self.data_dir,
                         "--language", "zh",
                         "--index", sparse_index_path,
                         "--generator", "DefaultLuceneDocumentGenerator",
                         "--threads", "1"]
        subprocess.run(["python", "-m", "pyserini.index.lucene"] + pyserini_args)
        print("Finish!")
        print(f"Sparse index path: {sparse_index_path}")
    
    def build_dense_index(self):
        """
        Building dense retrieval index based on faiss.

        Firstly, train the DAM module(if specified). Then use model to conver contents in 
        json files to embeddings. Finally, build faiss index based on embeddings.
        """

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_id)
        dense_index_path = self.index_save_dir + "/dense"
        faiss_save_path = dense_index_path + "/faiss.index"
        if not check_dir(dense_index_path):
            warnings.warn("Dense index already exists and will be overwritten.", UserWarning)
        
        # load json files
        print("Start loading data...")
        self.all_docs = load_json_data(self.data_dir)
        print("Finish.")
        
        # train DAM module if specified
        if self.train_dam_flag:
            print("Start training DAM...")
            self.dam_path = self.train_dam()
            print("Finish training!")
        else:
            print("Utilize a pre-existing, trained DAM")

        # convert doc to vector
        print("Start converting documents to vectors...")
        if self.use_content_type == "title":
            doc_content = [item['title'] for item in self.all_docs]
        elif self.use_content_type == "contents":
            doc_content = [item['contents'] for item in self.all_docs]
        else:
            doc_content = [item['title'] + item['contents'] for item in self.all_docs]
        model, tokenizer = self.build_model(dam_path = self.dam_path)
        model.cuda()
        doc_dataset = TextDataset(doc_content, tokenizer)
        doc_loader = torch.utils.data.DataLoader(doc_dataset, batch_size=self.batch_size, collate_fn=collate_fn)
        doc_embedding = []
        for batch in tqdm(doc_loader):
            batch = tuple(t.cuda() for t in batch)
            with torch.no_grad():
                output = model(input_ids = batch[0],attention_mask = batch[1])
            doc_embedding.append(output)
        doc_embedding = torch.cat(doc_embedding, dim=0)
        print("Finish converting embeddings.")

        # Build faiss index by using doc embedding
        print("Start building faiss index...")
        hidden_dim = doc_embedding.shape[1]
        dense_index = faiss.IndexFlatL2(hidden_dim)
        dense_index.add(doc_embedding.cpu().numpy())
        faiss.write_index(dense_index,faiss_save_path)
        print("Finish building index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Creating index based on JSON documents.")

    # Basic parameters
    parser.add_argument('--index_type', type=str, default="dense", choices=['sparse','dense','all'], 
                        help="Index types that need to be built.")
    parser.add_argument('--data_dir', type=str, required=True, 
                        help="The folder path for Json files.")
    parser.add_argument('--index_save_dir', type=str, required=True, 
                        help="Path to the folder where the index is stored.")
    
    # Parameters for building dense index
    parser.add_argument('--train_dam_flag', action='store_true', 
                        help="Train DAM based on data or use existing DAM modules.")
    parser.add_argument('--use_content_type', type=str, default='title', choices=['title','contents','all'],
                        help="The part of the document used to build an index.")
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128, 
                        help="Batch size used when constructing vector representations of documents")
    parser.add_argument('--dam_path', type=str, default="jingtao/DAM-bert_base-mlm-dureader", 
                        help="The path of the DAM to be used. This paramater will be used only when not training the DAM module.")
    args = parser.parse_args()

    index_builder = Index_Builder(index_type = args.index_type,
                                  data_dir = args.data_dir,
                                  index_save_dir = args.index_save_dir,
                                  train_dam_flag = args.train_dam_flag,
                                  use_content_type = args.use_content_type,
                                  cuda_id = args.cuda_id,
                                  batch_size = args.batch_size,
                                  dam_path = args.dam_path)
    index_builder.build_index()
