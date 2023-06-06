from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
from utils import get_argument_parser, Hard_Negative_Dataset, set_seed, get_dist_info
from models import ReGenBiEncoder

import warnings
warnings.filterwarnings("ignore", "Detected call of", UserWarning)

def main(args):
    set_seed(1234)
    rank, _ = get_dist_info()

    learning_rate_callback = LearningRateMonitor()
    process_bar_callback = TQDMProgressBar(refresh_rate=5)
    
    train_data = Hard_Negative_Dataset(args, args.train_file)
    val_data = Hard_Negative_Dataset(args, args.dev_file)
    net = ReGenBiEncoder(args, train_data, val_data) # TODO
    if rank == 0:
        print(f"initialized {args.rank}...")

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_path,
        verbose=True,
        every_n_epochs=1,
        monitor="val_loss",
        save_top_k=-1,
        mode="min",
        save_last=True
    )

    trainer = pl.Trainer(
        accelerator="auto",
        default_root_dir=args.output_path,
        gradient_clip_val=args.clip_grad_norm,
        max_epochs=args.epoch,
        strategy=DDPStrategy(find_unused_parameters=False),
        num_nodes=args.n_nodes,
        check_val_every_n_epoch=1,
        callbacks=[learning_rate_callback, checkpoint_callback, process_bar_callback],
        precision=16,
        log_every_n_steps=1,
    )
    trainer.fit(net)

if __name__ == "__main__":
    args = get_argument_parser()
    main(args)
