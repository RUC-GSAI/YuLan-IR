import argparse
import os
from tqdm import tqdm
import json
from transformers import AutoTokenizer
import torch

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch",
        help="epochs to train the model", type=int, default=40)
    parser.add_argument("-bs", "--batch-size", 
        help="batch size in training", type=int, default=32)
    parser.add_argument("-bse", "--batch-size-encode", 
        help="batch size in encoding", type=int, default=512)
    parser.add_argument("-ck","--checkpoint", 
        help="load the model from checkpoint before training/evaluating", type=str, default="")
    parser.add_argument("--dropout", 
        help="dropout probability", type=float, default=0.1)
    parser.add_argument("-lr", "--learning-rate", 
        help="learning rate", type=float, default=3e-6)
    parser.add_argument("--warmup", 
        help="warmup steps of scheduler", type=float, default=0.1)
    parser.add_argument("-cln", "--clip-grad-norm", 
        help="max norm of clipping gradients", type=float, default=0.0)
    parser.add_argument("-hits", 
        help="hit number per query", type=int, default=1000)
   
    parser.add_argument("-op", "--output-path", help="", type=str, default="./output/")
    parser.add_argument("-wd", "--weight-decay", help="weight decay of AdamW", type=float, default=0.01)
    parser.add_argument("-dr", "--data-root", type=str, default="data")

    # for splade model
    parser.add_argument("-si", "--splade-index", type=str, default="")
    parser.add_argument("-sop", "--splade-output-path", type=str, default="")
    parser.add_argument("-st", "--splade-tmp", type=str, default="")
    parser.add_argument("-qs", "--q_lambda", help="", type=float, default=1e-5)
    parser.add_argument("-ds", "--d_lambda", help="", type=float, default=1e-5)


    parser.add_argument("-ibs", "--index-batch-size", help="batch size for passage index", type=int, default=4)
    parser.add_argument("-ebs", "--encode-batch-size", help="batch size for encode query", type=int, default=16)

    parser.add_argument("-tf", "--train-file", type=str, default="train.tsv")
    parser.add_argument("-df", "--dev-file", type=str, default="dev.tsv")
    parser.add_argument("-cp", "--corpus-path", help="path to corpus", type=str, default="")

    parser.add_argument("-ql", "--query-length", help="query token length", type=int, default=16)
    parser.add_argument("-sl", "--sequence-length", help="sequence token length", type=int, default=256)
    parser.add_argument("-plm", help="short name of pre-trained language models", type=str, default="splade")
    parser.add_argument("-m", "--mode", help="train/test", type=str, default="train")

    parser.add_argument('--rank', type=int, default=0, help='Index of current task')  
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')  
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--init_method', default=None, help='print process when training')  

    args = parser.parse_args()
    if args.plm == "splade":
        args.splade_query_plm = os.path.join(args.data_root, "PLM", f"{args.plm}-query")
        args.splade_doc_plm = os.path.join(args.data_root, "PLM", f"{args.plm}-doc")
        args.tokenizer = AutoTokenizer.from_pretrained(args.splade_doc_plm)
    else:
        args.plm_dir = os.path.join(args.data_root, "PLM", args.plm)
        args.tokenizer = AutoTokenizer.from_pretrained(args.plm_dir)
    if args.splade_output_path and not os.path.exists(args.splade_output_path):
        os.makedirs(args.splade_output_path)

    args.n_nodes = args.world_size
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.world_size

    return args

if __name__ == "__main__":
    args = get_argument_parser()
    print(args)
