import os
import sys
import gzip
import random
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Optional, List, Dict
from dataclasses import dataclass, field

sys.path.append("..")
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import is_main_process
from transformers.data import DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    corpus_path: str = field()
    max_seq_length: int = field()
    

@dataclass
class ModelArguments:
    model_name_or_path: str = field()


class CorpusDataset(Dataset):
    def __init__(self, corpus: List[str], tokenizer: PreTrainedTokenizer, max_seq_length: int):
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_token_len = max_seq_length - 2
        # ignore warning about `Token indices sequence length is longer than the specified maximum sequence length for this model'
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item):
        if isinstance(self.corpus[item], str):
            text = self.corpus[item]
            cache_input_ids = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)['input_ids']
            self.corpus[item] = np.array(cache_input_ids, dtype=np.uint16) # cache
        
        input_ids = self.corpus[item].tolist()
        if len(input_ids) > self.max_token_len:
            start_pos = random.randint(0, len(input_ids)-self.max_token_len)
            input_ids = input_ids[start_pos: start_pos + self.max_token_len]
        
        batch_encoding = self.tokenizer.prepare_for_model(input_ids, add_special_tokens=True, return_special_tokens_mask=True)
        return batch_encoding


def read_corpus(file_path, verbose):
    logger.info(f"Load corpus: {file_path}")
    open_function = gzip.open if file_path.endswith(".gz") else open
    dataset = []
    for idx, line in tqdm(enumerate(open_function(file_path)), disable=not verbose, mininterval=10):
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        splits = line.split("\t")
        if len(splits) == 2:
            _id, text = splits
        else:
            raise NotImplementedError("Corpus Format: id\\ttext\\n")
        dataset.append(text)
    return dataset


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )
    
    resume_from_checkpoint = False
    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and any((x.startswith("checkpoint") for x in os.listdir(training_args.output_dir)))
    ):
        if not training_args.overwrite_output_dir:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        else:
            resume_from_checkpoint = True

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, config=config, use_fast=False)
    model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)
        
    train_set = CorpusDataset(
        read_corpus(data_args.corpus_path, verbose=is_main_process(training_args.local_rank)),
        tokenizer, data_args.max_seq_length
    )
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if is_main_process(training_args.local_rank):
        trainer.save_model()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
