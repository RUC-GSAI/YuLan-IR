import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
import random
import numpy as np
import os
import time
import sys
import math
import moxing as mox
from tqdm import tqdm
from torch.utils.data import DataLoader
from chunk_dataset import TextDataset
from bart_generation import FusionModel
from fid_model import FiDBART
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartModel, BartTokenizer, BartForConditionalGeneration
from torch.utils.data.distributed import DistributedSampler

mox.file.shift('os', 'mox')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="", type=str, help="")
    parser.add_argument("--config_name", default="", type=str, help="")
    parser.add_argument("--tokenizer_name", default="", type=str, help="")
    parser.add_argument("--data_dir", default="/cache/data/", type=str, help="")
    parser.add_argument("--output_dir", default="/cache/output/models/", type=str, help="")
    parser.add_argument("--result_dir", default="/cache/output/result/", type=str, help="")
    parser.add_argument("--pretrained_ckpt_path", default="/cache/pretrained_ckpt/", type=str, help="")

    parser.add_argument("--use_pretrained_ckpt", action="store_true", help="")
    parser.add_argument("--per_gpu_train_batch_size", default=50, type=int, help="")
    parser.add_argument("--per_gpu_eval_batch_size", default=50, type=int, help="")
    parser.add_argument("--max_encoding_length", default=256, type=int, help="")
    parser.add_argument("--max_decoding_length", default=512, type=int, help="")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="")
    parser.add_argument("--max_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=-1, type=int, help="")
    parser.add_argument("--logging_steps", default=100, type=int, help="")
    parser.add_argument("--save_steps", default=-1, type=int, help="")
    parser.add_argument("--seed", default=0, type=int, help="")
    parser.add_argument("--max_ref_num", default=5, type=int, help="")

    parser.add_argument("--save_ckpt_name", default="", type=str, help="")
    parser.add_argument("--pretrained_ckpt_name", default="", type=str, help="")
    parser.add_argument("--train_data_name", default="", type=str, help="")
    parser.add_argument("--test_data_name", default="", type=str, help="")
    parser.add_argument("--result_data_name", default="", type=str, help="")

    # ddp args
    parser.add_argument('--rank', type=int, default=0, help='Index of current task')  
    parser.add_argument('--world_size', type=int, default=1, help='Total number of tasks')  
    parser.add_argument('--dist_backend', type=str, default='nccl', help='distributed backend')
    parser.add_argument('--init_method', default=None, help='print process when training')  

    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    print("available GPU numbers:", ngpus_per_node)
    args.batch_size = args.per_gpu_train_batch_size
    args.test_batch_size = args.per_gpu_eval_batch_size

    tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.add_tokens("[title]")
    tokenizer.add_tokens("[ref]")
    tokenizer.add_tokens("[0]")
    tokenizer.add_tokens("[1]")
    tokenizer.add_tokens("[2]")
    tokenizer.add_tokens("[3]")
    tokenizer.add_tokens("[4]")
    tokenizer.add_tokens("[5]")
    args.tokenizer = tokenizer

    set_seed(args.seed)
    train_chunk_list = split_data(args)  # split into chunks

    # mp.spawn(train_model, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    daemon = False
    mp = multiprocessing.get_context('spawn')
    error_queues = []
    processes = []
    for i in range(ngpus_per_node):
        process = mp.Process(
            target=train_model,
            args=(i, ngpus_per_node, train_chunk_list[i + args.rank * ngpus_per_node], args),
            daemon=daemon,
        )
        process.start()
        processes.append(process)

    for p in processes:
        p.join()


def split_data(args):
    train_data_dir = args.data_dir + args.train_data_name
    train_data = []
    with open(train_data_dir, encoding="utf8") as src:
        train_data.extend(src.readlines())
    num_lines = len(train_data)

    num_replicas = args.world_size # node * gpu_per_node
    num_samples = int(math.ceil(num_lines * 1.0 / num_replicas))
    total_size = num_samples * num_replicas
    indices = list(range(num_lines)) # original dataset indices
    padding_size = total_size - len(indices)
    if padding_size <= len(indices):
        indices += indices[:padding_size]
    else: 
        indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

    train_chunk_list = []
    for i in range(num_replicas):
        train_chunk_list.append([train_data[item] for item in indices[i * num_samples:(i + 1) * num_samples]])
    ## check if the dimension is correct
    for ele in train_chunk_list:
        assert len(ele) == num_samples
    assert len(train_chunk_list) == num_replicas
    return train_chunk_list


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_model(local_rank, ngpus_per_node, train_data, args):
    args.local_rank = local_rank
    global_rank = args.rank * ngpus_per_node + local_rank
    print("rank: %s, local rank: %s." % (args.rank, args.local_rank))
    if local_rank is not None:
        print("Use GPU: {} for training".format(args.local_rank))
    print("backend:", args.dist_backend)
    print(args.dist_backend, args.init_method, args.world_size, global_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=global_rank)
    print("init finished")
    # test data
    test_data = []
    test_data_dir = args.data_dir + args.test_data_name
    with open(test_data_dir, encoding="utf8") as src:
        test_data.extend(src.readlines())
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart.resize_token_embeddings(len(args.tokenizer))
    if args.use_pretrained_ckpt:
        print("Loading pretrained model...")
        model_state_dict = torch.load(args.pretrained_ckpt_path + args.pretrained_ckpt_name, map_location='cpu')
        bart.load_state_dict(model_state_dict, strict=True)
    fid_bart = FiDBART(config)
    fid_bart.load_bart(bart.state_dict())
    model = FusionModel(fid_bart, config)
    if global_rank == 0:
        n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
        print("* number of parameters: %d" % n_params, flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    args.device = device
    model.to(args.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    # model = torch.nn.DataParallel(model)
    print("start fit")
    fit(model, train_data, test_data, args, global_rank)


def fit(model, X_train, X_test, args, global_rank):
    train_dataset = TextDataset(X_train, args.tokenizer, max_encoding_length=args.max_encoding_length, max_decoding_length=args.max_decoding_length, max_ref_num=args.max_ref_num)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs
    if args.save_steps < 0:
        args.save_steps = len(train_dataloader) // 3 - 1
    if args.warmup_steps < 0:
        args.warmup_steps = len(train_dataloader) // 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
    if global_rank == 0:
        time.sleep(1)
        print("***** Running Training *****", flush=True)
        print("Number of Epochs = ", args.num_train_epochs, flush=True)
        print("Examples per GPU =", len(train_dataset), flush=True)
        print("Total Examples = ", len(train_dataset) * args.world_size, flush=True)
        print("Batch Size per GPU = ", args.per_gpu_train_batch_size, flush=True)
        print("Total Train Batch Size = ", args.batch_size * args.world_size, flush=True)
        print("Steps per Epoch=", len(train_dataloader), flush=True)
        print("Total Optimization Steps = ", t_total, flush=True)

    best_result = 1e5
    global_step = 0
    for epoch in range(args.num_train_epochs):
        # train_sampler.set_epoch(epoch)
        if global_rank == 0:
            print("\nEpoch ", epoch + 1, "/", args.num_train_epochs, flush=True)
        model.train()
        total_loss = 0.0
        tmp_loss = 0.0
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            gen_loss = train_step(model, batch, args)
            gen_loss = gen_loss.mean()
            loss = gen_loss
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1
            if global_rank == 0 and step > 0:
                # print ever 100 local step
                if step % args.logging_steps == 0:
                    print("Step = {:d}\tLR = {:.6f}\tTotal Loss = {:.6f}\tTime Consumed = {:.2f}".format(step, scheduler.get_last_lr()[0], (total_loss - tmp_loss) / args.logging_steps, time.time() - start_time), flush=True)
                    tmp_loss = total_loss
                    start_time = time.time()
                # eval and save 3 times per epoch
                if step % args.save_steps == 0:
                    print("Step = {:d}\tLR = {:.6f}\tStart Evaluation".format(step, scheduler.get_last_lr()[0]), flush=True)
                    best_result = evaluate(model, X_test, best_result, args)
                    model.train()
            if args.max_steps > 0 and global_step > args.max_steps:
                break
            if global_step == 1 and global_rank == 0:
                os.system("nvidia-smi")
        if global_rank == 0:
            print("Epoch = {:d}\tLoss = {:.6f}".format(epoch + 1, total_loss / len(train_dataloader)), flush=True)
        if args.max_steps > 0 and global_step > args.max_steps:
            break


def train_step(model, train_data, args):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(args.device)
    gen_loss = model.forward(train_data)
    return gen_loss


def evaluate(model, X_test, best_result, args):
    model.eval()
    test_dataset = TextDataset(X_test, args.tokenizer, max_encoding_length=args.max_encoding_length, max_decoding_length=args.max_decoding_length, max_ref_num=args.max_ref_num)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, drop_last=True)
    all_test_loss = 0.0
    with torch.no_grad():
        for test_data in test_dataloader:
            for key in test_data.keys():
                test_data[key] = test_data[key].to(args.device)
            gen_loss = model.forward(test_data)
            all_test_loss += gen_loss.mean().item()
    all_test_loss = all_test_loss / len(test_dataloader)
    if all_test_loss < best_result:
        perplexity = torch.exp(torch.tensor(all_test_loss))
        print("Best Test Loss = {:.6f}, Perplexity = {:.6f}".format(all_test_loss, perplexity.item()), flush=True)
        best_result = all_test_loss
        model_to_save = model.module if hasattr(model, 'module') else model
        # model_to_save.model.save_pretrained(args.output_dir)
        torch.save(model_to_save.state_dict(), args.output_dir + args.save_ckpt_name)
    return best_result


if __name__ == '__main__':
    main()
