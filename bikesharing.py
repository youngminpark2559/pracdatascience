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

# load train data and test data file
train = pd.read_csv("D://chromedown//train.csv", parse_dates=["datetime"])
# print(train.shape)
# row, column (10886, 12)
test = pd.read_csv("D://chromedown//test.csv", parse_dates=["datetime"])
# print(test.shape)
# row, column (6493, 9)


# I want to process feature engineering
# I loaded train data and test data as datetime type
# I want to make them detailed data type
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second
train["dayofweek"] = train["datetime"].dt.dayofweek
# print(train.shape)
# (10886, 19)

test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["day"] = test["datetime"].dt.day
test["hour"] = test["datetime"].dt.hour
test["minute"] = test["datetime"].dt.minute
test["second"] = test["datetime"].dt.second
test["dayofweek"] = test["datetime"].dt.dayofweek
# print(test.shape)
# (6493, 16)

# I visualize data
# I can see there are a lot of data on windspeed 0
# It might come from a bad measurement so I need to refine them
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count',title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='Count',title="test windspeed")
sns.countplot(data=test, x="windspeed", ax=axes[1])



# I separate data into windspeed 0 and windspeed not 0 in train data
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
# I can see windspeed not 0 is much more many than windspeed 0
print(trainWind0.shape)
# (1313, 19)
print(trainWindNot0.shape)
# (9573, 19)


from sklearn.ensemble import RandomForestClassifier
def predict_windspeed(data):
    
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]
    
    # I select features for expecting windspeed model
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    # I change data type of windspeed not 0 data to the string to use
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    # I will use random forest classifier
    rfModel_wind = RandomForestClassifier()

    # I'm making a expecting model for the windspeed
    # And the training data for this expecting model is data composed of wCol
    # I want to find optimized parameters(letting an expecting model for the windspeed to learn) for this expecting model
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0["windspeed"])

    # Let the expecting model to expect the windspeed
    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])

    # New data holders
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0

    predictWind0["windspeed"] = wind0Values

    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다.
    data = predictWindNot0.append(predictWind0)

    # datatype of windspeed as float
    data["windspeed"] = data["windspeed"].astype("float")

    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data


# I use predict_windspeed() with putting train data to refine windspeed 0 data
train = predict_windspeed(train)
# test = predict_windspeed(test)

# I visualize refined data
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)

plt.sca(ax1)
# data label text on x axis is rotated by 30 degree
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='Count', title="windspeed of refined train data")
sns.countplot(data=train, x="windspeed", ax=ax1)
# I can confirm windspeed 0 data eleminated


I need to process "feature selection"
1. It's required to distinguish between meaningful data and noise
1. It doesn't mean the more feature, the better performance
1. It's recommended to add feature one by one with testing the performance and eliminate that feature if it turned out it's not that helpful feature
# continuous feature and categorical feature 
# continuous feature = ["temp","humidity","windspeed","atemp"]

# I choose following features as categorical feature
categorical_feature_names = ["season", "holiday", "workingday", "weather", "dayofweek", "month", "year", "hour"]
# categorical feature is needed to be a categorical data type
for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

# They show entire features
feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed", "year", "hour", "dayofweek", "holiday", "workingday"]
print(feature_names)
# ['season',
#  'weather',
#  'temp',
#  'atemp',
#  'humidity',
#  'windspeed',
#  'year',
#  'hour',
#  'dayofweek',
#  'holiday',
#  'workingday']

# I create a new matrix X_train after pre processing above
X_train = train[feature_names]
# print(X_train.shape)
# (10886, 11)
# X_train.head()
# table




plt.show()