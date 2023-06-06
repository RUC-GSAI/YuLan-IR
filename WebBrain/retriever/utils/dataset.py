from torch.utils.data import Dataset
import os 
import json

class Hard_Negative_Dataset(Dataset):
    def __init__(self, args, data_path) -> None:
        """
        iterably load the triples, tokenize and return
        """
        self.args = args
        super().__init__()

        self.query_length = args.query_length
        self.sequence_length = args.sequence_length
        self.corpus = {}
        for line in open(args.corpus_path):
            line = json.loads(line)
            self.corpus[line["id"]] = line["contents"]

        self.data = []
        for i,line in enumerate(open(data_path)):
            line = line.strip().split("\t")
            self.data.append(line)
        
        self.tokenizer = args.tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, n):
        line = self.data[n]
        query = line[1]
        pos_neg_seqs_ids = line[2:]
        pos_neg_seqs = [self.corpus[int(k)] for k in pos_neg_seqs_ids]

        query_output = self.tokenizer(
            query, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.query_length, 
            truncation=True
            )

        pos_neg_seqs_output = self.tokenizer(
            pos_neg_seqs, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.sequence_length, 
            truncation=True
            )

        return_dict = {
            "q_feats": query_output,
            "pos_neg_feats": pos_neg_seqs_output,
        }
        return return_dict

class Text_Dataset(Dataset):
    def __init__(self, args, loaded_data=None, is_query=False):
        super().__init__()
        self.tokenizer = args.tokenizer
        if not is_query:
            self.max_length = args.sequence_length
        else:
            self.max_length = args.query_length
        if loaded_data is None:
            self.data_path = os.path.join(args.data_root, args.corpus_path)
            self.data = open(self.data_path).read().strip().split('\n')
        else:
            self.data= loaded_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        feat = self.tokenizer(self.data[item][1], 
                                return_tensors="pt", 
                                padding="max_length", 
                                max_length=self.max_length, 
                                truncation=True)
        feat["id"] = item
        return feat
