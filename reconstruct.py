import os
import torch
import numpy as np
import os, sys
from tqdm import tqdm
from model import FCModel
from utils.GMM_utils import getTest, makeGMM
from utils import gumbel_sinkhorn_ops
sys.path.append(os.pardir)

block_size = 8
loaded_model = './docs/Nyx_8.pth'
# loaded_model = './log/Nyx_8.pth'
testdir = f'./GMMDATA/testdataset/test'
Datasize = (np.array([64, 64, 64])//block_size)*block_size

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"


def make_prob(log_alpha, samples, xyz, hist_array, xmin, xmax):
    ### sinkhorn prob
    binsize = 128
    tau, n_sink_iter = 5, 30
    prob = gumbel_sinkhorn_ops.gumbel_sinkhorn(torch.Tensor(log_alpha), tau, n_sink_iter)
    
    sam = samples[0].cpu().detach().numpy()
    trans = prob[0].cpu().detach().numpy().transpose()

    for i in range(block_size):
        for j in range(block_size):
            for k in range(block_size):
                n = i*block_size*block_size + j*block_size + k
                a, b = zip(*sorted(zip(sam, trans[n]), key=lambda x: x[0]))

                i_, j_, k_ = xyz + [i, j, k]
                hist, bins = np.histogram(a, bins=binsize, weights=b, density=True, range=(xmin, xmax))
                prob2 = hist* np.diff(bins)

                hist_array[i_][j_][k_] = prob2

def recon(num_workers, model):
    allfile = os.listdir(testdir)

    for f in tqdm(allfile):
        if f.startswith('x'): continue
        fileparam = f[2:-4]
        test_data = getTest(os.path.join(testdir, f), num_workers)

        # these two variables are use for isosurface
        # xmin, xmax = np.inf, np.NINF
        # hist_array = [[[0 for _ in range(Datasize[2])] for __ in range(Datasize[1])] for ___ in range(Datasize[0])]

        result1 = np.zeros(Datasize, np.float32)
        result2 = np.zeros(Datasize, np.float32)

        ### first iteration to create NaiveBD
        for param, y_, xyz in tqdm(test_data):
            x, y, z = [t.numpy().astype(int) for t in xyz[0]]

            gmm = makeGMM(np.array(y_[0]))
            try:
                samples = torch.Tensor(gmm.sample(block_size**3)[0]).view(1, block_size**3).to(device)
            except:
                idx = gmm.weights_.argmax()
                gmm.weights_[idx] -= np.finfo(np.float32).epsneg
                samples = torch.Tensor(gmm.sample(block_size**3)[0]).view(1, block_size**3).to(device)

            shuffle_idx = torch.randperm(n_numbers)
            samples = samples.view(-1)[shuffle_idx].view(1, block_size**3)

            block = samples.cpu().data.view(block_size, block_size, block_size).numpy()

            result1[x:x+block_size, y:y+block_size, z:z+block_size] = block
        result1.tofile(f'./result/naiveBD/{fileparam}_resample.bin')


        for param, y_, xyz in tqdm(test_data):
            x, y, z = [t.numpy().astype(int) for t in xyz[0]]
            param = param.to(device)
            
            gmm = makeGMM(np.array(y_[0]))
            try:
                samples = torch.Tensor(gmm.sample(block_size**3)[0]).view(1, block_size**3).to(device)
            except:
                idx = gmm.weights_.argmax()
                gmm.weights_[idx] -= np.finfo(np.float32).epsneg
                samples = torch.Tensor(gmm.sample(block_size**3)[0]).view(1, block_size**3).to(device)

            shuffle_idx = torch.randperm(n_numbers)
            samples = samples.view(-1)[shuffle_idx].view(1, block_size**3)

            # xmin = min(samples.min().item(), xmin)
            # xmax = max(samples.max().item(), xmax)


            X = samples.detach().clone()
            X -= X.min(1, keepdim=True)[0]
            X /= X.max(1, keepdim=True)[0]

            xyz = torch.LongTensor(xyz.long()).to(device)

            log_alpha = model(X, xyz, param)
            assingment_matrix = gumbel_sinkhorn_ops.gumbel_matching(log_alpha, noise=False)

            est_permutation = assingment_matrix.max(1)[1].int()
            est_sample = samples[:, est_permutation]

            block = est_sample.cpu().data.view(block_size, block_size, block_size).numpy()
            # # if you want to create histogram for isosurface
            # make_prob(log_alpha, samples, xyz.cpu().numpy()[0], hist_array, xmin, xmax)

            result2[x:x+block_size, y:y+block_size, z:z+block_size] = block

        # # create histogram for calcualting isosurface
        # np.savez(f'./log/hist/{fileparam}.npz', min=xmin, max=xmax, hist=np.array(hist_array, dtype=object))
        result2.tofile(f'./result/reconstruct/{fileparam}.bin')

if __name__=='__main__':
    num_workers = 8
    batch_size = 1
    hid_c = 768
    n_numbers = block_size**3

    model = FCModel(batch_size, hid_c, n_numbers).to(device)
    # CPU
    # model.load_state_dict(torch.load(loaded_model, map_location='cpu'), strict=False)
    
    # GPU
    model.load_state_dict(torch.load(loaded_model, weights_only=True), strict=False)
    model.eval()

    recon(num_workers, model)
