import numpy as np,glob
from scipy.signal import detrend
import pandas as pd
import os,time
from config import DATA_ROOT_PATH,FEATURE_PATH
from feature_extraction.get_time_feature import extraction_time_feature
from feature_extraction.get_frequency_feature import extraction_frequency_feature
from feature_extraction.get_entropy_feature import extraction_entropy_feature

# 为不同时间滑动窗口设计

# data loading
path = os.path.abspath(os.path.join(DATA_ROOT_PATH,"./Bern Barcelona/ALL/*.txt"))
files = sorted(glob.glob(path))

# init
FS = []

FS_all = []
segments = []

# 采样率
fs = 512
# 时间窗移动
overlap = 0
# 片段时长
segments_time = 5

# 遍历文件
for i in range(len(files)):
    print("===== Feature extracting {:d}  =====".format(i))
    X = pd.read_csv(files[i],sep=",",header=None)
    # 相邻导联相减
    all_data = (X[0] - X[1]).values

    # 时间窗移动 求解特征
    j = 0
    count = 0
    FS_single = []
    time_elapsed_tf = 0; time_elapsed_ff = 0; time_elapsed_ef = 0
    while (j < len(all_data)) & (j+segments_time*fs <= len(all_data)): # 时间窗的坐标索引
        segments = all_data[j:j+segments_time*fs]
        data = detrend(segments) # 数据去趋势

        since_tf = int(time.time() * 1000)
        time_feature_segments = extraction_time_feature(data)
        time_elapsed_tf += int(time.time() * 1000) - since_tf
        time_feature_segments = np.array(time_feature_segments).T

        since_ff = int(time.time() * 1000)
        frequency_feature_segments = extraction_frequency_feature(data,fs)
        time_elapsed_ff += int(time.time() * 1000) - since_ff
        frequency_feature_segments = np.array(frequency_feature_segments).T

        since_ef = int(time.time() * 1000)
        entropy_feature_segments = extraction_entropy_feature(data,fs)
        time_elapsed_ef += int(time.time() * 1000) - since_ef
        entropy_feature_segments = np.array(entropy_feature_segments).flatten()
        # 时间窗坐标索引
        j = j + int(segments_time*fs*(1-overlap))
        # 片段个数
        count = count + 1
        # 合并每个片段的时域特征、频域特征、时频域特征和非线性熵特征
        FS = np.hstack((time_feature_segments,frequency_feature_segments, entropy_feature_segments))
        # 合并整条信号不同片段的特征
        FS_single = np.hstack((FS_single,FS))
    # 将路径内的文件 特征求解
    print('The Time feature extraction code run {:.0f}ms'.format(time_elapsed_tf))
    print('The Frequency feature extraction code run {:.0f}ms'.format(time_elapsed_ff))
    print('The Entropy feature extraction code run {:.0f}ms'.format(time_elapsed_ef))
    print('The Feature extraction code run {:.0f}ms'.format(time_elapsed_tf + time_elapsed_ff + time_elapsed_ef))
    FS_single = np.array(FS_single)
    FS_all.append(FS_single)
FS_all = np.array(FS_all)
print(FS_all.shape)
np.savetxt(os.path.join(FEATURE_PATH,"Bern_classical_features_20s.csv"), FS_all, delimiter=",")
print("All data Feature Finished successfully.")