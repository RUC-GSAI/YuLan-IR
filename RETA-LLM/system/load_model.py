from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from typing import List, Optional, Tuple, Union
import torch
import json
from searcher import Dense_Searcher
from answer_generator import Answer_Generator
from request_rewriter import Request_Rewriter
from passage_extractor import Passage_Extractor
from fact_checker import Fact_Checker
from config import *

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"

@st.cache_resource
def get_model():
    '''
    Prepare the LLM, searcher and other modules.
    '''
    ### Load your own LLM (api) by editing this
    if ("chatgpt" in model_config_path):
        model = "chatgpt"
    
    if ("chatglm" in model_config_path) :
        model_config = json.load(open(model_config_path,"r"))
        model_path = model_config['model_path']
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
        model = model.eval()

    if ("llama" in model_config_path or "yulan" in model_config_path) :
        model_config = json.load(open(model_config_path,"r"))
        model_path = model_config['model_path']
        print(f'Loading {model_path}...')

        tokenizer_init_kwargs = model_config.get('tokenizer_init_kwargs', dict())
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_init_kwargs)
        if 'Linly' in model_path or '65b' in model_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
        model_init_kwargs = model_config.get('model_init_kwargs', dict())
        if isinstance(model_config["device"], str):
            print(model_config.get("causal_lm", True))
            if model_config.get("causal_lm", True):
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_init_kwargs)
            else:
                model = AutoModel.from_pretrained(model_path, **model_init_kwargs)
            device = torch.device(model_config["device"])
            model = model.eval().half().to(device)
        else:
            config = AutoConfig.from_pretrained(model_path, **model_init_kwargs)
            with init_empty_weights():
                if model_config.get("causal_lm", True):
                    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, **model_init_kwargs)
                else:
                    model = AutoModel.from_config(config, torch_dtype=torch.float16, **model_init_kwargs)
            model.tie_weights()
            load_checkpoint_and_dispatch_kwargs = model_config.get('load_checkpoint_and_dispatch_kwargs', dict())
            model = load_checkpoint_and_dispatch(
                model, model_path, device_map=model_config["device"], dtype=torch.float16, **load_checkpoint_and_dispatch_kwargs
            )
        
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        else:
            model.config.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    ###


    passage_extractor = Passage_Extractor(model, tokenizer)

    fact_checker = Fact_Checker(model, tokenizer)

    searcher = Dense_Searcher()
    
    request_rewriter = Request_Rewriter(model, tokenizer)
    
    answer_generator = Answer_Generator(model, tokenizer)

    return request_rewriter, searcher, passage_extractor, answer_generator, fact_checker
