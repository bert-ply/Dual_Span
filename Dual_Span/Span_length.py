# -*- coding: utf-8 -*-
# @Time : 2023/5/22 22:30
# @Author : Pan Li
import matplotlib.pyplot as plt
import numpy
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

span_length = [1, 2, 3, 4, 5, 6, 7, 8,9]
x = range(len(span_length))
Lap14 = [53.57, 58.08, 63.39, 63.58, 63.03, 62.8, 62.67, 64.58, 59.12]
Res14 = [63.36, 73.6, 75.04, 74.38, 74.93, 71.63, 75.01, 75.38, 67.23]
Res15 = [58.84, 64.72, 61.52, 65.98, 65.72, 62.96, 63.23, 67.13, 62.52]
Res16 = [62.42, 70.1, 70.76, 67.9, 66.59, 66.18, 73.19, 73.66, 64.36]

# plt.plot(x, ACC, color ='red', marker='o', linestyle='-', label='ACC', linewidth=0.8)
# plt.plot(x, F1, color='red', marker='D', linestyle='-', label='M-F1', linewidth=0.8)
plt.plot(x, Lap14, color='red', marker='o', linestyle='-', label='Lap14', linewidth=1, markersize=10)
plt.plot(x, Res14, color='green', marker='^', linestyle='-', label='Res14', linewidth=1, markersize=10)
plt.plot(x, Res15, color='blue', marker='*', linestyle='-', label='Res15', linewidth=1, markersize=10)
# plt.plot(x, BERT_ACC_RGAT, color='red', marker='o', linestyle='-', label='W/O_APRC_ACC', linewidth=0.8)
plt.plot(x, Res16, color='purple', marker='D', linestyle='-', label='Res16', linewidth=1, markersize=10)
# plt.plot(x, BERT_F1_RGAT, color='red', marker='D', linestyle='-', label='W/O_APRC_F1', linewidth=0.8)
plt.legend(loc="lower right", prop={"size": 10})
plt.xticks(x, span_length, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Maximum Span Length", fontsize=20)
plt.ylabel("F1(%)", fontsize=20)
plt.grid(linestyle=':',linewidth=0.5)
# plt.title('Restaurant',fontsize=20)
#Laptop,Restaurant,MAMS
plt.show()
plt.savefig('layers.jpg')