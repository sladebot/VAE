from pytorch_lightning import Trainer
from torchvision.utils import save_image
from models import vae_models
from config import config
from PIL import Image 
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from torch.nn.functional import interpolate
from torchvision.transforms import Resize, ToPILImage, Compose
from torchvision.utils import make_grid

def load_model(ckpt, model_type="vae"):
    model = vae_models[model_type].load_from_checkpoint(f"./saved_models/{ckpt}")
    model.eval()
    return model

def parse_model_file_name(file_name):
    # Hard Coded Parsing based on the filenames that I use
    substrings = file_name.split(".")[0].split("_")
    name, alpha, dim = substrings[0], substrings[2], substrings[4]
    new_name = ""
    if name == "vae":
        new_name += "Vanilla VAE"
    new_name += f" | alpha={alpha}"
    new_name += f" | dim={dim}"
    return new_name

def tensor_to_img(tsr):
    if tsr.ndim == 4:
        tsr = tsr.squeeze(0)
    
    transform = Compose([
        ToPILImage()
    ])
    img = transform(tsr)
    return img


def resize_img(img, w, h):
    return img.resize((w, h))


def export_to_onnx(ckpt):
    model = load_model(ckpt)
    filepath = "model.onnx"
    test_iter = iter(model.test_dataloader())
    sample, _ = next(test_iter)
    model.to_onnx(filepath, sample, export_params=True)
