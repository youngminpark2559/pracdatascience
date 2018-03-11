# @Chapter
# 자전거 수요 예측[3/4] 캐글 머신러닝 랜덤포레스트만으로 경진대회에 참여하기
# https://www.youtube.com/watch?v=g7EwIFXJntc&t=179s

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

# load train data file
train = pd.read_csv("D://chromedown//train.csv", parse_dates=["datetime"])
print(train.shape)