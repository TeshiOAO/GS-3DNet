import os, sys
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from utils.GMM_utils import my_data_load
from model import FCModel
from utils import gumbel_sinkhorn_ops
sys.path.append(os.pardir)

def train(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    if torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.benchmark = True
    else:
        device = "cpu"
    
    model = FCModel(cfg.batch_size, cfg.hid_c, cfg.n_numbers).to(device)
    optimizer = optim.Adam(model.parameters(), cfg.lr)

    ### plot
    loss_plt = []

    train_loader = my_data_load('./GMMDATA/testdataset/train', cfg.batch_size, cfg.num_workers)

    for epoch in range(cfg.epochs):
        # training do not need gmm information
        for param, x_batch, y_batch, xyz in tqdm(train_loader):
            param = param.to(device)

            ordered_X = x_batch.detach().clone().to(device)
            ordered_X -= ordered_X.min(1, keepdim=True)[0]
            ordered_X /= ordered_X.max(1, keepdim=True)[0]

            X = ordered_X[:, torch.randperm(ordered_X.size()[1])].to(device)

            xyz = torch.LongTensor(xyz.type(torch.int64)).to(device)

            log_alpha = model(X, xyz, param)

            gumbel_sinkhorn_mat = [
                gumbel_sinkhorn_ops.gumbel_sinkhorn(log_alpha, cfg.tau, cfg.n_sink_iter)
                for _ in range(cfg.n_samples)
            ]

            est_ordered_X = [
                gumbel_sinkhorn_ops.inverse_permutation(X, gs_mat)
                for gs_mat in gumbel_sinkhorn_mat
            ]

            loss = sum([
                torch.nn.functional.mse_loss(X, ordered_X)
                for X in est_ordered_X
            ]).to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{epoch} epoch training loss {loss.item():.6f}")
        loss_plt.append(loss.item())

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"./log/chk/chk{epoch}.pth")

    
    torch.save(model.state_dict(), "./log/Nyx_8.pth")
    plt.clf()
    plt.plot(loss_plt)
    plt.savefig('./log/img/Nyx_8.png')



if __name__ == "__main__":
    import argparse
    param_grid ={
        'tau': [10],
        'n_sink_iter': [50],
        'n_samples': [3],
        'lr': [0.005], 
    }
    blocksize = 8

    best_score = torch.inf
    best_hyperparams = {}

    hyp_param = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

    parser = argparse.ArgumentParser()
    # gumbel sinkhorn option
    parser.add_argument("--tau", default=hyp_param['tau'], type=float, help="temperture parameter")
    parser.add_argument("--n_sink_iter", default=hyp_param['n_sink_iter'], type=int, help="number of iterations for sinkhorn normalization")
    parser.add_argument("--n_samples", default=hyp_param['n_samples'], type=int, help="number of samples from gumbel-sinkhorn distribution")
    # datase option
    parser.add_argument("--n_numbers", default=blocksize**3, type=int, help="number of sorted numbers")
    parser.add_argument("--num_workers", default=8, type=int, help="number of threads for CPU parallel")
    # optimizer option
    parser.add_argument("--lr", default=hyp_param['lr'], type=float, help="learning rate")
    parser.add_argument("--batch_size", default=32, type=int, help="mini-batch size")
    parser.add_argument("--epochs", default=40, type=int, help="number of epochs")
    # misc
    parser.add_argument("--hid_c", default=768, type=int, help="number of hidden channels")
    parser.add_argument("--out_dir", default="log", type=str, help="/path/to/output directory")
    parser.add_argument("--data", default="GMM8", type=str, help="select the data for training")

    cfg = parser.parse_args()

    if not os.path.exists(cfg.out_dir):
        os.mkdir(cfg.out_dir)

    # logger setup
    logging.basicConfig(
        filename=os.path.join(cfg.out_dir, "console.log"),
    )
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    logger = logging.getLogger("NumberSorting")
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    train(cfg)