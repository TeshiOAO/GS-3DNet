import os
import torch

import torch.utils.data as Data
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

# this file is used to load data and create GMM samples

def my_data_load(Path, batch_size, num_workers):
    allfile = os.listdir(Path)
    param = []
    x_data, y_data, xyz = [], [], []
    for f in allfile:
        file = np.load(os.path.join(Path,f))
        if f.startswith('y'):
            param.extend(file['p'])
            xyz.extend(file['xyz'])
            means = file['means']
            covs = file['cov']
            weights = file['weight']
            y_data.extend(zip(means, covs, weights))
        elif f.startswith('x'):
            x_data.extend(file['data'])

    Param = torch.Tensor(np.array(param))
    x_data, y_data, xyz = torch.Tensor(np.array(x_data)), torch.Tensor(np.array(y_data)), torch.Tensor(np.array(xyz))
    torch_dataset = Data.TensorDataset(Param, x_data, y_data, xyz)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return loader

def getTest(Path, num_workers):
    file1 = np.load(Path)
    param = []
    y_data, xyz = [], []
    
    param.extend(file1['p'])
    xyz.extend(file1['xyz'])
    means = file1['means']
    covs = file1['cov']
    weights = file1['weight']
    y_data.extend(zip(means, covs, weights))

    Param = torch.Tensor(np.array(param))
    y_data, xyz = torch.Tensor(np.array(y_data)), torch.Tensor(np.array(xyz))
    torch_dataset = Data.TensorDataset(Param, y_data, xyz)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=1,
        num_workers=num_workers
    )
    return loader

def makeGMM(y_data):
    means, covs, weight = [np.array(d) for d in y_data]
    gmm = GMM(n_components=5)
    gmm.means_ = means.reshape(5, 1)
    gmm.weights_ = weight
    gmm.covariances_ = covs.reshape(5, 1, 1)
    gmm.precisions_cholesky_ = _compute_precision_cholesky(gmm.covariances_, 'full')

    return gmm