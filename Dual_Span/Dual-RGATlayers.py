# -*- coding: utf-8 -*-
# @Time : 2023/5/25 15:03
# @Author : Pan Li
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

span_length = [1, 2, 3, 4, 5, 6, 7, 8]
x = range(len(span_length))
Lap14 = [63.5,	64.58,	63.2,	63.82,	61.67,	62.9,	61.61,	61.18]
Res14 = [71.21,	75.38,	72.36,	72.79,	71.87,	72.54,	71.29,	70.97]
Res15 = [63.82,	67.13,	61.93,	65.3,	64.64,	65.1,	64.09,	63.70]
Res16 = [69.75,	73.66,	70.37,	71.24,	70.67,	71.35,	70.86,	69.17]

# plt.plot(x, ACC, color ='red', marker='o', linestyle='-', label='ACC', linewidth=0.8)
# plt.plot(x, F1, color='red', marker='D', linestyle='-', label='M-F1', linewidth=0.8)
plt.plot(x, Lap14, color='red', marker='o', linestyle='-', label='Lap14', linewidth=1, markersize=10)
plt.plot(x, Res14, color='green', marker='^', linestyle='-', label='Res14', linewidth=1, markersize=10)
plt.plot(x, Res15, color='blue', marker='*', linestyle='-', label='Res15', linewidth=1, markersize=10)
# plt.plot(x, BERT_ACC_RGAT, color='red', marker='o', linestyle='-', label='W/O_APRC_ACC', linewidth=0.8)
plt.plot(x, Res16, color='purple', marker='D', linestyle='-', label='Res16', linewidth=1, markersize=10)
# plt.plot(x, BERT_F1_RGAT, color='red', marker='D', linestyle='-', label='W/O_APRC_F1', linewidth=0.8)
plt.legend(loc="upper right", prop={"size": 10})
plt.xticks(x, span_length, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("DualRGAT layers", fontsize=20)
plt.ylabel("F1(%)", fontsize=20)
plt.grid(linestyle=':',linewidth=0.5)
# plt.title('Restaurant',fontsize=20)
#Laptop,Restaurant,MAMS
plt.show()
plt.savefig('layers.jpg')