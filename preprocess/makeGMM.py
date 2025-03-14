import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from sklearn.mixture import GaussianMixture as GMM
import random

datatype = {
    'Nyx'       : ( 64,  64,  64,),#(256, 256, 256,),
    'RedSea'    : ( 50, 500, 500,),
}


blocksize = 8

# Set target data and save location
dname = 'Nyx'
Path = './preprocess/makeGMM_input'
savedir= './preprocess/makeGMM_output'

target = 'density1.bin'

allFolderList = [f for f in os.listdir(Path)]
allFileList = []
for folder in allFolderList:
    p = Path+"/"+folder
    FileList = [p+"/"+f for f in os.listdir(p)]
    allFileList += FileList

print('Start to Transform from raw_Nyx to GMM(mean, covariance, weight):')
for file in tqdm(allFileList):
    # Extracting file name for processing
    filepath = file + "/" + target
    fs = file.split("/")
    Pram, Time = fs[3].split("_"), fs[4][3:]
    # print(Pram, Time)

    outx = os.path.join(savedir, 'x_'+ Pram[0] + '_' + Pram[1] + '_' + Pram[2] + '_' + Time + '.npz')
    outy = os.path.join(savedir, 'y_'+ Pram[0] + '_' + Pram[1] + '_' + Pram[2] + '_' + Time + '.npz')

    data = np.fromfile(filepath, dtype='float32')
    data = data.reshape(datatype[dname])

    P = []
    T = []
    XYZ = []
    Mean = []
    Cov = []
    Weight = []
    bdata = []

    for i in range(0, datatype[dname][0], blocksize):
        for j in range(0, datatype[dname][1], blocksize):
            for k in range(0, datatype[dname][2], blocksize):
                xmax = min(i+blocksize, datatype[dname][0])
                ymax = min(j+blocksize, datatype[dname][1])
                zmax = min(k+blocksize, datatype[dname][2])
                block_ = data[i:xmax, j:ymax, k:zmax]
                block = block_.reshape(-1,1)
                n_components = 5
                best_gmm = GMM(n_components=n_components, max_iter=1000)
                best_gmm.fit(block)

                xyz = [i,j,k]

                P.append(Pram)
                T.append(np.array([Time]))
                XYZ.append(xyz)
                Mean.append(best_gmm.means_.flatten())
                Cov.append(best_gmm.covariances_.flatten())
                Weight.append(best_gmm.weights_.flatten())
                bdata.append(block)
    P = np.array(P, dtype=np.float32)
    T = np.array(T, dtype=np.float32)
    XYZ = np.array(XYZ, dtype=np.float32)
    Mean = np.array(Mean, dtype=np.float32)
    Cov = np.array(Cov, dtype=np.float32)
    Weight = np.array(Weight, dtype=np.float32)
    bdata = np.array(bdata, dtype=np.float32)

    np.savez(outy, p=P, t=T, xyz=XYZ, means=Mean, cov=Cov, weight=Weight)
    np.savez(outx, data=bdata)