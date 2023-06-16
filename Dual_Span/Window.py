# -*- coding: utf-8 -*-
# @Time : 2023/5/23 15:15
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

window = [1, 2, 3, 4]
x = range(len(window))
Lap14 = [62.26,64.49,64.58, 61.5 ]
Res14 = [72.89, 73.17, 75.38,70.78]
Res15 = [65.12, 63.12, 67.13, 63.57]
Res16 = [71.09, 71.87,73.66, 69.97]

# plt.plot(x, ACC, color ='red', marker='o', linestyle='-', label='ACC', linewidth=0.8)
# plt.plot(x, F1, color='red', marker='D', linestyle='-', label='M-F1', linewidth=0.8)
plt.plot(x, Lap14, color='red', marker='o', linestyle='-', label='Lap14', linewidth=1, markersize=10)
plt.plot(x, Res14, color='green', marker='^', linestyle='-', label='Res14', linewidth=1, markersize=10)
plt.plot(x, Res15, color='blue', marker='*', linestyle='-', label='Res15', linewidth=1, markersize=10)
# plt.plot(x, BERT_ACC_RGAT, color='red', marker='o', linestyle='-', label='W/O_APRC_ACC', linewidth=0.8)
plt.plot(x, Res16, color='darkorange', marker='D', linestyle='-', label='Res16', linewidth=1, markersize=10)
# plt.plot(x, BERT_F1_RGAT, color='red', marker='D', linestyle='-', label='W/O_APRC_F1', linewidth=0.8)
plt.legend(loc="upper right", prop={"size": 10})
plt.xticks(x, window, fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Window", fontsize=20)
plt.ylabel("F1(%)", fontsize=20)
plt.grid(linestyle=':',linewidth=0.5)
# plt.title('Restaurant',fontsize=20)
#Laptop,Restaurant,MAMS
plt.show()
plt.savefig('layers.jpg')