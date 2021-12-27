from pytorch_lightning import Trainer
from models import vae_models
from config import config
from pytorch_lightning.loggers import TensorBoardLogger
import os

def make_model(config):
    model_type = config.model_type
    model_config = config.model_config

    if model_type not in vae_models.keys():
        raise NotImplementedError("Model Architecture not implemented")
    else:
        return vae_models[model_type](**model_config.dict())


if __name__ == "__main__":
    model_type = config.model_type
    model = vae_models[model_type].load_from_checkpoint("./saved_models/vae_alpha_1024_dim_128.ckpt")
    logger = TensorBoardLogger(**config.log_config.dict())
    trainer = Trainer(gpus=1, logger=logger)
    trainer.test(model)