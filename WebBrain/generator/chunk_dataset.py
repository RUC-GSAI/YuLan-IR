from torch.utils.data import Dataset
import numpy as np

class TextDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_encoding_length, max_decoding_length, max_ref_num):
        super(TextDataset, self).__init__()
        self._data = data_list
        self._tokenizer = tokenizer
        self._max_encoding_length = max_encoding_length
        self._max_decoding_length = max_decoding_length
        self._max_ref_num = max_ref_num
        self._total_data = len(data_list)

    def __getitem__(self, idx):
        line = self._data[idx]
        line = line.strip().split("[SPLIT]")

        target = line[-1]
        source = line[:-1]
        source = source[:self._max_ref_num]

        all_source_input_ids = []
        all_source_attention_mask = []
        for source_input in source:
            source_encode = self._tokenizer(source_input, padding="max_length", max_length=self._max_encoding_length, truncation=True)
            source_input_ids = source_encode.input_ids
            source_attention_mask = source_encode.attention_mask
            all_source_input_ids.append(source_input_ids)
            all_source_attention_mask.append(source_attention_mask)
        while len(all_source_input_ids) < self._max_ref_num:
            source_encode = self._tokenizer("", padding="max_length", max_length=self._max_encoding_length, truncation=True)
            source_input_ids = source_encode.input_ids
            source_attention_mask = source_encode.attention_mask
            all_source_input_ids.append(source_input_ids)
            all_source_attention_mask.append(source_attention_mask)

        target_encode = self._tokenizer(target, padding="max_length", max_length=self._max_decoding_length, truncation=True)
        target_input_ids = target_encode.input_ids
        target_labels = np.asarray(target_input_ids)
        target_labels[target_labels == self._tokenizer.pad_token_id] = -100
        target_remark_labels = target_labels.copy()
        target_remark_labels = target_remark_labels - 50267
        target_remark_labels[target_remark_labels < 0] = -100

        batch = {
            "input_ids": np.asarray(all_source_input_ids, dtype=np.int64),
            "attention_mask": np.asarray(all_source_attention_mask, dtype=np.int64),
            "target_labels": np.asarray(target_labels, dtype=np.int64),
            "target_remark_labels": np.asarray(target_remark_labels, dtype=np.int64)
        }

        return batch

    def __len__(self):
        return self._total_data
