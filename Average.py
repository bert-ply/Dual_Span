# -*- coding: utf-8 -*-
# @Time : 2023/5/22 22:29
# @Author : Pan Li
import pandas as pd

#计算5次平均值

seed = []
df = pd.read_csv('result.csv')
print(df)
df = df.sort_values(by=['F'])
df = df.iloc[-5:, :]
print(df)
print(df.mean())
seed.extend(list(df['S']))
print(seed)