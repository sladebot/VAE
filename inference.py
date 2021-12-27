from models import vae_models
from config import config
from PIL import Image 
from torchvision.transforms import Resize, ToPILImage, Compose

from utils import load_model, tensor_to_img, resize_img, export_to_onnx



def predict(model_ckpt="vae_alpha_1024_dim_128.ckpt"):
    model_type = config.model_type
    model = vae_models[model_type].load_from_checkpoint(f"./saved_models/{model_ckpt}")
    model.eval()
    test_iter = iter(model.test_dataloader())
    d, _ = next(test_iter)
    _, _, out = model(d)
    out_img = tensor_to_img(out)
    return out_img

    

if __name__ == "__main__":
    predict()
    # export_to_onnx("./saved_models/vae.ckpt")