import torch
import torch.nn as nn
import torch.nn.init as init


class FusionModel(nn.Module):
    def __init__(self, bart, config):
        super(FusionModel, self).__init__()
        self.fidbart = bart
        self.config = config
    
    def forward(self, batch_data):
        """
        Args: context: [batch, 2, len]
        """
        input_ids = batch_data["input_ids"]
        attention_mask = batch_data["attention_mask"]
        target_labels = batch_data["target_labels"]

        seq_outputs = self.fidbart(input_ids=input_ids, attention_mask=attention_mask, labels=target_labels, output_attentions=True, output_hidden_states=True, return_dict=True)
        gen_loss = seq_outputs.loss

        return gen_loss
