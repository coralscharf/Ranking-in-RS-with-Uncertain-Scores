import numpy as np
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar, EarlyStopping


class LitProgressBar(TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm(disable=True)
        return bar


def train(model, data, path, name):
    prog_bar = LitProgressBar()
    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=False, mode='min')
    cp = ModelCheckpoint(monitor='val_loss', dirpath=path, filename=name+'-{epoch}-{val_loss:.4f}', save_weights_only=True)
    trainer = Trainer(min_epochs=20, max_epochs=200, logger=False, callbacks=[prog_bar, es, cp])  # gpus=1, Coral's Update
    trainer.fit(model, datamodule=data)
