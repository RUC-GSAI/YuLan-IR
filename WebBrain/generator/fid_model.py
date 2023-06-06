import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class FiDBART(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()
    
    def wrap_encoder(self):
        self.model.encoder = EncoderWrapper(self.model.encoder)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            if input_ids.dim() == 3:
                self.model.encoder.n_cands = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def unwrap_encoder(self):
        self.model.encoder = self.model.encoder.encoder

    def load_bart(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.model.encoder.n_cands = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            **kwargs
        )


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.n_cands = 0
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs,):
        bsz, total_length = input_ids.shape  # total_length = n_cands * cand_length
        cand_length = total_length // self.n_cands
        input_ids = input_ids.view(bsz * self.n_cands, cand_length)
        attention_mask = attention_mask.view(bsz * self.n_cands, cand_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(bsz, self.n_cands * cand_length, -1) # fid
        return outputs