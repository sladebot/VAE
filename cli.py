import click
import yaml
from box import Box
from os import path
from util import get_config
from main import train, evaluate


@click.command()
@click.argument('cmd')
@click.option('--config', '-c', default="./configs/core.yml")
def main(cmd, config):
    if config is None:
        raise Exception("No config file provided")
    cfg = get_config(config)

    # TODO: Add CLI overrides
    if cmd == "train":
        print("train")
        train(cfg.train)
    elif cmd == "test":
        evaluate(cfg.test)
    else:
        raise Exception("Unknown mode")

if __name__ == "__main__":
    main()


# # --- parsing and configuration --- #
# parser = argparse.ArgumentParser(
#     description="PyTorch implementation of VAE for MNIST")
# parser.add_argument("train", type=bool, default=True, help="Run training")
# # parser.add_argument("test", type=bool, default=False, help="Run training")
# parser.add_argument('--config', type=str, default="configs/core.yml",
#                     help='configuration file config/*.yml')    
# parser.add_argument('--batch-size', type=int, default=128,
#                     help='batch size for training (default: 128)')
# parser.add_argument('--epochs', type=int, default=20,
#                     help='number of epochs to train (default: 20)')
# parser.add_argument('--z-dim', type=int, default=2,
#                     help='dimension of hidden variable Z (default: 2)')
# parser.add_argument('--log-interval', type=int, default=100,
#                     help='interval between logs about training status (default: 100)')
# parser.add_argument('--learning-rate', type=int, default=1e-3,
#                     help='learning rate for Adam optimizer (default: 1e-3)')
# parser.add_argument('--prr', type=bool, default=True,
#                     help='Boolean for plot-reproduce-result (default: True')
# parser.add_argument('--prr-z1-range', type=int, default=2,
#                     help='z1 range for plot-reproduce-result (default: 2)')
# parser.add_argument('--prr-z2-range', type=int, default=2,
#                     help='z2 range for plot-reproduce-result (default: 2)')
# parser.add_argument('--prr-z1-interval', type=int, default=0.2,
#                     help='interval of z1 for plot-reproduce-result (default: 0.2)')
# parser.add_argument('--prr-z2-interval', type=int, default=0.2,
#                     help='interval of z2 for plot-reproduce-result (default: 0.2)')

# args = parser.parse_args()
# cfg = get_config(args.config)

# if args.train:
#     train(cfg.train)
#     print(cfg.train)

# # pin memory provides improved transfer speed
# kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

# train_loader = torch.utils.data.DataLoader(train_data,
#                                            batch_size=BATCH_SIZE, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(test_data,
#                                           batch_size=BATCH_SIZE, shuffle=True, **kwargs)





# def test(epoch):
#     model.eval()
#     test_loss = 0
#     with torch.no_grad():
#         for batch_idx, (data, label) in enumerate(test_loader):
#             data = data.to(device)
#             recon_data, mu, logvar = model(data)
#             cur_loss = loss_function(recon_data, data, mu, logvar).item()
#             test_loss += cur_loss

#             if batch_idx == 0:
#                 # saves 8 samples of the first batch as an image file to compare input images and reconstructed images
#                 num_samples = min(BATCH_SIZE, 8)
#                 comparison = torch.cat(
#                     [data[:num_samples], recon_data.view(BATCH_SIZE, 1, 28, 28)[:num_samples]]).cpu()
#                 save_generated_img(
#                     comparison, 'reconstruction', epoch, num_samples)

#     test_loss /= len(test_loader.dataset)
#     print('====> Test set loss: {:.4f}'.format(test_loss))

# def sample_from_model(epoch):
#     with torch.no_grad():
#         # p(z) = N(0,I), this distribution is used when calculating KLD. So we can sample z from N(0,I)
#         sample = torch.randn(64, Z_DIM).to(device)
#         sample = model.decode(sample).cpu().view(64, 1, 28, 28)
#         save_generated_img(sample, 'sample', epoch)

# def plot_along_axis(epoch):
#     z1 = torch.arange(-Z1_RANGE, Z1_RANGE, Z1_INTERVAL).to(device)
#     z2 = torch.arange(-Z2_RANGE, Z2_RANGE, Z2_INTERVAL).to(device)
#     num_z1 = z1.shape[0]
#     num_z2 = z2.shape[0]
#     num_z = num_z1 * num_z2

#     sample = torch.zeros(num_z, 2).to(device)

#     for i in range(num_z1):
#         for j in range(num_z2):
#             idx = i * num_z2 + j
#             sample[idx][0] = z1[i]
#             sample[idx][1] = z2[j]

#     sample = model.decode(sample).cpu().view(num_z, 1, 28, 28)
#     save_generated_img(sample, 'plot_along_z1_and_z2_axis', epoch, num_z1)


# if args.train:
#     train()

# # if __name__ == '__main__':
# #     for epoch in range(1, EPOCHS + 1):
# #         train(epoch)
# #         test(epoch)
# #         sample_from_model(epoch)

# #         if PRR:
# #             plot_along_axis(epoch)