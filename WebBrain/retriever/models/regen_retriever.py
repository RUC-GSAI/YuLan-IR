
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup
from utils import get_dist_info
from models import ranking_loss

class ReGenLightning(pl.LightningModule):
    def __init__(self, plm_dir, cp=None, out_type="sparse", agg="max"):
        super(ReGenLightning, self).__init__()
        self.encoder = AutoModelForMaskedLM.from_pretrained(plm_dir)
        if cp:
            print("loading from ", cp)
            states = torch.load(cp).state_dict()
            self.encoder.load_state_dict(states)
        self.out_type = out_type

        assert agg in ("sum", "max")
        self.agg = agg

    def forward(self, feats):
        seq_length = feats["input_ids"].size()[-1]
        feats = {k:v.view(-1, seq_length) for k,v in feats.items()}
        out = self.encoder(**feats)["logits"]
        if self.agg == "sum":
            values = torch.sum(torch.log(1 + torch.relu(out)) * feats["attention_mask"].unsqueeze(-1), dim=1)
        else:
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * feats["attention_mask"].unsqueeze(-1), dim=1)
        return values

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        idx = batch.pop("id").tolist()
        emb = self(batch)
        if self.out_type == "dense":
            return emb, idx
        elif self.out_type == "sparse":
            row, col = torch.nonzero(emb, as_tuple=True)
            data = emb[row, col]
            return row, col, data, idx

class ReGenBiEncoder(pl.LightningModule):
    def __init__(self, 
                 args,
                 train_data=None,
                 dev_data=None):

        super(ReGenBiEncoder, self).__init__()
        self.args = args
        self.train_data = train_data
        self.dev_data = dev_data
        self.query_encoder = ReGenLightning(self.args.splade_query_plm)
        self.doc_encoder = ReGenLightning(self.args.splade_doc_plm)

        if self.args.checkpoint:
            states = torch.load(self.args.checkpoint)["state_dict"]
            self.load_state_dict(states)
            print(f"loaded from {self.args.checkpoint}...")
        
    def forward(self, batch):
        q_vectors = self.query_encoder(batch["q_feats"])  
        bs, n_docs, _ = batch["pos_neg_feats"]["input_ids"].size()
        doc_vectors = self.doc_encoder(batch["pos_neg_feats"])
        doc_vectors = doc_vectors.view(bs, n_docs, -1)
        pos_doc_vectors = doc_vectors[:,0,:]
        neg_doc_vectors = doc_vectors[:,1:,:]
        loss = ranking_loss(q_vectors, pos_doc_vectors, neg_doc_vectors)
        q_l1_loss = torch.sum(torch.abs(q_vectors*self.args.q_lambda), dim=-1).mean()
        d_l1_loss = torch.sum(torch.abs(doc_vectors*self.args.d_lambda), dim=-1).mean()
        loss = loss + q_l1_loss + d_l1_loss
        return loss

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_data)
        return DataLoader(
            self.train_data,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True, sampler=train_sampler
        )

    def val_dataloader(self):
        return DataLoader(
            self.dev_data,
            batch_size=self.args.batch_size,
            num_workers=4
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        total_training_step = int(len(self.train_data)*self.args.epoch/self.args.batch_size)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            int(total_training_step*self.args.warmup),
             total_training_step
             )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]


    def training_step(self, batch, batch_nb):
        loss = self(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True
        )

        return loss

    def validation_step(self, batch, batch_nb):
        loss = self(batch)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True
        )
        return loss

    def validation_epoch_end(self, outputs):
        outputs = [l.detach().item() for l in outputs]
        outputs = sum(outputs) / len(outputs)
        self.log(
            "val loss",
            outputs,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True
        )
        rank, word_size = get_dist_info()
        if rank == 0:
            torch.save(
                self.query_encoder.encoder, 
                os.path.join(self.args.splade_output_path, f'query_epoch_{self.current_epoch}.pt')
            )

            torch.save(
                self.doc_encoder.encoder, 
                os.path.join(self.args.splade_output_path, f'doc_epoch_{self.current_epoch}.pt')
            )

