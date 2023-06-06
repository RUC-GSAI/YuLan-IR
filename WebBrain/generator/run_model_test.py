import argparse
import torch
import random
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from text_dataset import TextDataset
from bart_generation import FusionModel
from fid_model import FiDBART
from transformers import AdamW, get_linear_schedule_with_warmup, BartConfig, BartModel, BartTokenizer, BartForConditionalGeneration

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", default="", type=str, help="")
parser.add_argument("--config_name", default="", type=str, help="")
parser.add_argument("--tokenizer_name", default="", type=str, help="")
parser.add_argument("--data_dir", default="", type=str, help="")
parser.add_argument("--result_dir", default="", type=str, help="")
parser.add_argument("--max_encoding_length", default=256, type=int, help="")
parser.add_argument("--max_decoding_length", default=512, type=int, help="")
parser.add_argument("--max_ref_num", default=5, type=int, help="")
parser.add_argument("--seed", default=0, type=int, help="")
parser.add_argument("--ckpt_dir", default="", type=str, help="")

parser.add_argument("--save_ckpt_name", default="", type=str, help="")
parser.add_argument("--test_data_name", default="", type=str, help="")
parser.add_argument("--result_data_name", default="", type=str, help="")

args = parser.parse_args()

# need to modify
tokenizer = BartTokenizer.from_pretrained(args.tokenizer_name)
tokenizer.add_tokens("[title]")
tokenizer.add_tokens("[ref]")
tokenizer.add_tokens("[0]")
tokenizer.add_tokens("[1]")
tokenizer.add_tokens("[2]")
tokenizer.add_tokens("[3]")
tokenizer.add_tokens("[4]")
tokenizer.add_tokens("[5]")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(args)

def set_seed(seed=args.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def test_model_generation():
    test_data = args.data_dir + args.test_data_name
    config = BartConfig.from_pretrained(args.config_name)
    bart = BartForConditionalGeneration.from_pretrained(args.model_name_or_path, config=config)
    bart.resize_token_embeddings(len(tokenizer))
    fid_bart = FiDBART(config)
    fid_bart.load_bart(bart.state_dict())
    model = FusionModel(fid_bart, config)
    model_state_dict = torch.load(args.ckpt_dir + args.save_ckpt_name)
    model.load_state_dict(model_state_dict, strict=True)
    model = model.to(device)
    model.eval()

    test_dataset = TextDataset(test_data, tokenizer, max_encoding_length=args.max_encoding_length, max_decoding_length=args.max_decoding_length, max_ref_num=args.max_ref_num)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

    count = 0
    with open(args.result_dir + args.result_data_name, "w", encoding="utf-8") as fw:
        with torch.no_grad():
            for test_data in test_dataloader:
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
                outputs = model.fidbart.generate(
                    input_ids=test_data["input_ids"],
                    attention_mask=test_data["attention_mask"],
                    max_length=512,
                    # min_length=256,
                    no_repeat_ngram_size=3,
                    do_sample=False,
                    num_beams=5,
                    output_attentions=True,
                    output_hidden_states=True
                    # top_k=40,
                    # top_p=0.9
                )
                outputs = outputs.cpu()
                batch_out_sentences = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for idx, r in enumerate(batch_out_sentences):
                    fw.write(r + "\n")
                count += 4


if __name__ == '__main__':
    set_seed(args.seed)
    test_model_generation()
