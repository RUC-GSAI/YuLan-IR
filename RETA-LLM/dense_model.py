import torch
from torch import Tensor
import torch.nn.functional as F
from adaptertransformers.src.transformers.adapters import BertAdapterModel


SIMILARITY_METRIC_IP = "ip"
SIMILARITY_METRIC_COS = "cos"
SIMILARITY_METRICS = [SIMILARITY_METRIC_IP, SIMILARITY_METRIC_COS]

POOLING_AVERAGE = "average"
POOLING_CLS = "cls"
POOLING_METHODS = [POOLING_AVERAGE, POOLING_CLS]


def extract_text_embed(
        last_hidden_state: Tensor, 
        attention_mask: Tensor, 
        similarity_metric: str, 
        pooling: str
    ):
    if pooling == POOLING_CLS:
        text_embeds = last_hidden_state[:, 0]
    elif pooling == POOLING_AVERAGE:
        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
        text_embeds = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    else:
        raise NotImplementedError()
    if similarity_metric == SIMILARITY_METRIC_IP:
        pass
    elif similarity_metric == SIMILARITY_METRIC_COS:
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
    else:
        raise NotImplementedError()
    return text_embeds
#define BertDense Model
class BertDense(BertAdapterModel):
    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, return_dict=False):
        outputs = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            token_type_ids = token_type_ids, 
            position_ids = position_ids,
            return_dict = True,
        )
        pooling = getattr(self.config, "pooling")
        similarity_metric = getattr(self.config, "similarity_metric")
        text_embeds = extract_text_embed(
            last_hidden_state = outputs.last_hidden_state, 
            attention_mask = attention_mask,
            similarity_metric = similarity_metric, 
            pooling = pooling,
        )
        if return_dict:
            outputs.embedding = text_embeds
            return outputs
        else:
            return text_embeds