import yaml
from box import Box
from os import path
import torch

def get_config(config_path):
    if not path.exists(config_path):
        raise Exception(f"Config file not found, path: {config_path}")
    with open(config_path, 'r') as f:
        try:
            config = Box(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)
    return config

def get_device(logger):
    if torch.cuda.is_available():
        logger.info("GPU Available")
    else:
        logger.info("Running on CPU")
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
    return device

def load_checkpoint(checkpoint_file_path, cfg):
    """Loads model, optimizer from checkpoint file
    Args:
        checkpoint_file_path (str): Path to .pth checkpoint file
    Returns:
        model, optimizer, identifier
    """
    state_dict = torch.load(
        checkpoint_file_path,
        map_location=torch.device('cpu')
    )

    model_state_dict = state_dict["model_state_dict"]
    optimizer = state_dict["optimizer_state_dict"]

    model = DeepRNN(
        cfg.rnn_hidden_size,
        cfg.rnn_layers,
        cfg.rnn_dropout
    )

    model.load_state_dict(model_state_dict)

    return model, optimizer