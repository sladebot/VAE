import os
import torch
import torch.onnx
from torch.autograd import Variable
import torch.nn as nn
from util import get_device, load_checkpoint
from logger import MetricsClient

class VAEEvaluator:
    def __init__(
        self,
        checkpoint,
        cfg,
        criterion=None):

        # self.logger = setup_console_logger()
        self.device = get_device(self.logger)
        model, optimizer = load_checkpoint(checkpoint, cfg)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        self.model = model.to(self.device)
        self.metrics_client = MetricsClient(
            id=id,
            model=self.model,
            optimizer=optimizer,
            batch_size=cfg.batch_size,
            trajectory_len=cfg.trajectory_len,
            learning_rate=cfg.lr,
            checkpoint_path=checkpoint,
            cfg=cfg)
    
    def evaluate_epoch(self, input_tensor, epoch):
        self.logger.info(f"Starting inference")
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for 

        