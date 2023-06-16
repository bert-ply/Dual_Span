# -*- coding: utf-8 -*-
# @Time : 2023/4/12 10:05
# @Author : Pan Li
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib as mpl
from transformers import BertTokenizer

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False
#laptop
# column_labels = ["The","baterry","is","very","longer"]
# column_labels = ["And","windows","7","works","like","a","charm"]
# column_labels = ['i', 'wanted', 'it', 'for', 'it', "'", 's', 'mobility', 'and', 'man', ',', 'this', 'little', 'bad', 'boy', 'is', 'very', 'nice']

# sentence =[The baterry is very longer]
# res15
column_labels = ['good', 'creative', 'rolls']
# column_labels = ['the', 'wine', 'list', 'was', 'extensive', '-', 'though', 'the', 'staff', 'did', 'not', 'seem', 'knowledge', '##able', 'about', 'wine', 'pairing', '##s']
# column_labels =['tasty', 'delicious', 'food']
# sentence = ['for', '7', 'years', 'they', 'have', 'put', 'out', 'the', 'most', 'ta', '##sty', ',', 'most', 'delicious', 'food', 'and', 'kept', 'it', 'that', 'way']
model_path = "./My_BERT_Model"
tokenizer = BertTokenizer.from_pretrained(model_path)
print(tokenizer)
# sentnece = "Good creative rolls !"
# tokens = tokenizer.tokenize(sentnece)
# print(tokens)
#
# index = []
# for i in range(len(column_labels)):
#     print(column_labels[i])
#     if column_labels[i] in sentence:
#         print(sentence.index(column_labels[i]))
#         index.append(sentence.index(column_labels[i]))
# print(index)

# laptop
# df0 = pd.read_csv('./Attention/attention_B_syn75.csv')
# df0 = pd.read_csv('./Attention/attention_B_pos153.csv')
df0 = pd.read_csv('./Attention/attention_B_pos134.csv')

# print(max)
# print(df0)

#
df0 = df0.iloc[:len(column_labels), :len(column_labels)]
#
# df0 = df0.loc[[9, 10, 13, 14], ["9", "10", "13", "14"]]
# print(df0)
# #pos
data =[[0.26398385 ,0.31216994, 0.16067086],
 [0.21887696,0.2057213 , 0.12345498],
 [0.23884368, 0.30116376 ,0.4599926 ]]

# data = [[0.09007016,0.07966467,	0.108468955],
# [0.097491855,0.11626162,0.23953798],
# [0.109241785,0.09282404,0.09807052]]
#syn
# data = [[0.3129125,0.0000000,	0.2958995],
# [0.000000,0.338418,0.281782],
# [0.1486305,0.086461 ,0.130387]]
data = np.array(data)
min = data.min()
max = data.max()
print(min)
print(data)
# x_nor = (data - data.min(axis=0))/(data.max(axis=0)-data.min(axis=0))
# x_nor = pd.DataFrame(x_nor)
# 行归一化
# def norm(df0):
#     data = np.array(df0)
#     np1 = data.max(axis=1)
#     np2 = data.min(axis=1)
#     for i in range(len(data)):
#         for j in range(len(data[i])):
#             data[i][j] = (data[i][j] - np2[i])/(np1[i]-np2[i])
#     return data




# print(x_nor)
# for i in range(len(data)):
#     print(data.iloc[i].sum())
# print(x_nor)
# [18, 19, 20, 21, 31, 35, 37]
# df0 = x_nor.loc[index,  index]
# print(df0)
# df0 =[[  0.120400, 0.865806,  0.146738],
# [0.301364, 0.902564,  0.215590 ],
# [  0.080273  ,1.0 , 0.013835]]

# data = np.array(df0)


fig, ax = plt.subplots()
cmap = mpl.cm.cool
heatmap = ax.pcolor(data, cmap='YlGnBu', vmin=0, vmax=max)
heatmap.cmap.set_under('white')
# bar = fig.colorbar(heatmap, extend='both')
ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

ax.invert_yaxis()
ax.xaxis.tick_top()
# column_labels sentence
ax.set_xticklabels(column_labels, minor=False, fontsize=22,rotation=15,weight='bold',fontproperties='Times New Roman')
ax.set_yticklabels(column_labels, minor=False, fontsize=22,rotation=45,weight='bold',fontproperties='Times New Roman')

plt.show()