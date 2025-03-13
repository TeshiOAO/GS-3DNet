import os
import numpy as np
from tqdm import tqdm

blocksize = 8
resample = False
rawdata_shape = [64, 64, 64]
data_shape = (np.array(rawdata_shape)//blocksize)*blocksize

# data1 raw data, data2 reconstruction data
def PSNR(data1, data2):
    # 計算均方誤差（MSE）
    mse = np.mean((data1 - data2) ** 2)

    # 計算PSNR
    if mse == 0:
        psnr = float('inf')  # 如果MSE為0，PSNR理論上是無窮大（即兩個數據完全相同）
    else:
        Max = max(data1.flatten())
        psnr = 20 * np.log10(Max / np.sqrt(mse))

    return psnr


def SSIM(data1, data2):
    # Calculate mean of data1 and data2
    mean1 = np.mean(data1)
    mean2 = np.mean(data2)
    # Calculate variance of data1 and data2
    var1 = np.var(data1)
    var2 = np.var(data2)
    # Calculate covariance of data1 and data2
    cov = np.cov(data1.flatten(), data2.flatten())[0][1]
    # Constants for SSIM calculation
    c1 = (0.01 * np.max(data1)) ** 2
    c2 = (0.03 * np.max(data1)) ** 2
    # Calculate SSIM
    numerator = (2 * mean1 * mean2 + c1) * (2 * cov + c2)
    denominator = (mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2)
    ssim = numerator / denominator
    return ssim

def RMSE(data1, data2):
    # 計算均方根誤差（RMSE）
    rmse = np.sqrt(np.mean((data1 - data2) ** 2))
    return rmse


rawfilepath = r"./result/ground truth"           # the path of raw data
comparefilepath = r"./result/reconstruct"       #the path of reconstruction data

# Computing MSE, PSNR, and SSIM
PSNR_list, SSIM_list, RMSE_list = [], [], []

cmpfolder = os.listdir(comparefilepath)
if resample: cmpfolder = list(filter(lambda x: len(x.split('_')) == 5, cmpfolder))
else: cmpfolder = list(filter(lambda x: len(x.split('_')) == 4, cmpfolder))

for cmpf in tqdm(cmpfolder):
    rdf = cmpf
    if resample: rdf = rdf[:-13]+'.bin'
    file1 = os.path.join(rawfilepath, rdf)
    file2 = os.path.join(comparefilepath, cmpf)

    raw_data = np.fromfile(file1, dtype='float32').reshape(rawdata_shape)[:data_shape[0], :data_shape[1], :data_shape[2]]
    compare_data = np.fromfile(file2, dtype='float32').reshape(data_shape)

    PSNR_Score = PSNR(raw_data, compare_data)
    # print(f"PSNR: {PSNR_Score} dB")
    PSNR_list.append(PSNR_Score)

    SSIM_Score = SSIM(raw_data, compare_data)
    # print(f"SSIM: {SSIM_Score}")
    SSIM_list.append(SSIM_Score)

    RMSE_Score = RMSE(raw_data, compare_data)
    # print(f"RMSE: {RMSE_Score}")
    RMSE_list.append(RMSE_Score)

print("AVG_PSNR", np.average(PSNR_list))
print("AVG_SSIM", np.average(SSIM_list))
print("AVG_RMSE", np.average(RMSE_list))
