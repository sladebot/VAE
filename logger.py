import os
import torch
import logging
import google.cloud.logging
from tensorboardX import SummaryWriter
from google.cloud.storage import Client

FLOAT = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetricsClient:
    def __init__(self,
                 id,
                 model,
                 optimizer,
                 batch_size,
                 learning_rate,
                 checkpoint_path,
                 cfg,
                 summary_path="runs/",
                 logger=None):
        self.id = id
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.logger = logger
        
        self.writer = SummaryWriter(log_dir=self.summary_path)

        if logger is None:
            self.logger = setup_console_logger()
        self.gcs_client = Client()
        self.cfg = cfg
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

    def save_summary(self, data, summary_type="scalar", input_dims=None):
        if summary_type == "scalar":
            key, loss, idx = data
            self.writer.add_scalar(key, loss, idx)
            self.logger.info(f"Saved scalar - {key}")
            self.writer.flush()
        elif summary_type == "graph":
            sample_input = torch.rand(input_dims).to(device)
            self.writer.add_graph(data, sample_input)
            self.logger.info("Saved graph")
            self.writer.flush()
        elif summary_type == "custom":
            key, custom_loss, idx = data
            self.writer.add_scalars(key, custom_loss, idx)
            self.writer.flush()

    def save_checkpoint(self, counter, upload=False):
        filename = 'checkpoint_{}.pth'.format(counter)
        torch.save({
            'identifier': self.id,
            'counter': counter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'trajectory_size': self.trajectory_len
        }, os.path.join(self.checkpoint_path, filename))
        if upload:
            self.upload_checkpoint(filename)
        return filename

    def end_epoch(self, idx):
        self.logger.info(f"Finished epoch {idx}, identifier: {self.id}")
        self.writer.close()

    def upload_checkpoint(self, filename):
        source = os.path.join(self.checkpoint_path, filename)
        dest = f"{self.cfg.gcs_bucket}/{self.id}/checkpoints/{filename}"
        upload_blob(self.cfg.gcs_bucket, source, dest)
        self.logger.info(f"Checkpoint uploaded to GCS at {dest}.")

    def clear_disk(self, filename):
        os.remove(os.path.join(self.checkpoint_path, filename))


def setup_console_logger():
    '''
    Console logger
    '''
    # logging config
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(levelname)s | %(asctime)s | %(name)s | %(threadName)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_logger = logging.getLogger(__name__)
    return console_logger
