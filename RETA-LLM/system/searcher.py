import json
from pyserini.search.lucene import LuceneSearcher
import faiss
import torch
import numpy as np
from config import *
from model_response import generate_response
import os

import sys
sys.path.append("..")
from dense_model import *
from transformers import AutoConfig, AutoTokenizer, AutoModel
from adaptertransformers.src.transformers import PretrainedConfig

# This is the searcher module, we provide simple bm25-based sparse searcher and disentangled_retriever based denser searcher

class Common_Searcher:
    def __init__(self) :
        #initialize your customized searcher
        pass
    def search(self, query:str) -> list(dict()) :
        #insert your customized searcher's search function.
        #The result should be a list of dictionary.
        #The dictionary should at least contain these keys: title, contents, url.
        pass

class Dense_Searcher:
    def __init__(self):
        self.topk = topk
        # prepare encoder

        self.model, self.tokenizer = self.get_model()
        self.device = torch.device("cuda")
        self.model.to(self.device)

        # prepare reference docs
        self.all_doc = self.get_doc_from_folder()
        
        # prepare faiss index
        self.index = faiss.read_index(DENSE_INDEX_PATH)

    
    def get_model(self):
        config = PretrainedConfig.from_pretrained(DAM_NAME)
        config.similarity_metric, config.pooling = "ip", "average"
        tokenizer = AutoTokenizer.from_pretrained(DAM_NAME, config=config)
        model = BertDense.from_pretrained(DAM_NAME, config=config)
        adapter_name = model.load_adapter(REM_URL)
        model.set_active_adapters(adapter_name)
        model.eval()
        return model,tokenizer

    def get_doc_from_folder(self):
        """
        get all docs & properties from a folder containing json file
        """
        files = os.listdir(DOC_PATH)
        files.sort()
        all_data = []
        for file in files:
            print("Loading: ",file)
            file_path = os.path.join(DOC_PATH, file)
            with open(file_path, "r", encoding = "utf-8") as f:
                doc = json.load(f)
            all_data.append(doc)
        return all_data

    def search(self, query):
        tokenized_text = self.tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
        input_ids = tokenized_text["input_ids"].to(self.device)
        attention_mask = tokenized_text["attention_mask"].to(self.device)

        query_embed = self.model(input_ids = input_ids, attention_mask = attention_mask)
        query_embed = query_embed.detach().cpu().numpy()
        _,doc_id = self.index.search(query_embed, self.topk)
        doc_id = doc_id[0]  

        cnt = 0
        raw_reference_list = []
        for i,id in enumerate(doc_id):
            raw_reference = self.all_doc[id]        
            raw_reference_list.append(raw_reference)
    
            cnt += 1
            if (cnt == self.topk) :
                break

        return raw_reference_list

class Sparse_Searcher:
    def __init__(self):
        self.topk = topk
        self.all_doc = self.get_doc_from_folder()
        self.all_id_list = [item['id'] for item in self.all_doc]
        self.searcher = LuceneSearcher(SPARSE_INDEX_PATH)
        self.searcher.set_language(sparse_language)
    
    def get_doc_from_folder(self):
        """
        get all docs & properties
        """
        files = os.listdir(DOC_PATH)
        files.sort()
        all_data = []
        for file in files:
            print("Loading: ",file)
            file_path = os.path.join(DOC_PATH, file)
            with open(file_path, "r", encoding = "utf-8") as f:
                doc = json.load(f)
            all_data.append(doc)
        return all_data

    def search(self, query):
        hits = self.searcher.search(query)

        raw_reference_list = []

        cnt = 0       
        for i in range(min(self.topk, len(hits))):
            doc_id = hits[i].docid
            doc_index = self.all_id_list.index(doc_id)
            
            raw_reference = self.all_doc[doc_index]
            raw_reference_list.append(raw_reference)

            cnt += 1
            if (cnt == self.topk) :
                break

        return raw_reference_list