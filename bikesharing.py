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
# fig, axes = plt.subplots(nrows=2)
# fig.set_size_inches(18,10)

# plt.sca(axes[0])
# plt.xticks(rotation=30, ha='right')
# axes[0].set(ylabel='Count',title="train windspeed")
# sns.countplot(data=train, x="windspeed", ax=axes[0])

# plt.sca(axes[1])
# plt.xticks(rotation=30, ha='right')
# axes[1].set(ylabel='Count',title="test windspeed")
# sns.countplot(data=test, x="windspeed", ax=axes[1])



# I separate data into windspeed 0 and windspeed not 0 in train data
trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
# I can see windspeed not 0 is much more many than windspeed 0
# print(trainWind0.shape)
# (1313, 19)
# print(trainWindNot0.shape)
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


# I need to process "feature selection"
# 1. It's required to distinguish between meaningful data and noise
# 1. It doesn't mean the more feature, the better performance
# 1. It's recommended to add feature one by one with testing the performance and eliminate that feature if it turned out it's not that helpful feature
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
# print(feature_names)
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

# I create a new matrix X_test after pre processing above
X_test = test[feature_names]
# print(X_test.shape)
# (6493, 11)
# X_test.head()
# table

# I use count feature as y data
label_name = "count"
y_train = train[label_name]
# print(y_train.shape)
# (10886,)
# y_train.head()
# 0     1
# 1    36
# 2    56
# 3    84
# 4    94
# Name: count, dtype: int64


# "bike sharing contest" is evaluated by RMSLE

# I implement RMSLE algorithm in rmsle()
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values):
    # I will use data as numpy array
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # I should implement this formular
    # \sqrt{\frac{1}{n} \sum\limits_{i=1}^{n}(\log{(p_{i}+1)}-\log{(a_{i}+1)})^{2}}
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    difference = log_predict - log_actual
    # difference = (log_predict - log_actual) ** 2
    difference = np.square(difference)
    
    mean_difference = difference.mean()
    
    score = np.sqrt(mean_difference)
    
    return score

rmsle_scorer = make_scorer(rmsle)
# print(rmsle_scorer)


# I will use KFold for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)


# I will expect by random forest
from sklearn.ensemble import RandomForestRegressor
max_depth_list = []
# n_estimators higher makes better precision but consuming more time to expect
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)
# print(model)

# I get the cross validation score 
score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
score = score.mean()
# the closer to the 0 of score, it means better data
# print("Score= {0:.5f}".format(score))


# I make it learn by inputting feature(X_train), label(the answer, y_train)
model.fit(X_train, y_train)

# I make it predict based on learnt model by inputting X_test
predictions = model.predict(X_test)

# print(predictions.shape)
# (6493,)
print(predictions[0:10])
# array([12.43, 5.07, 4.44, 3.65, 3.2, 6.38, 38.77, 104.51,  235.12,  135.72])


# I visualize predicted data
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
ax1.set(title="train data of x features and y label")
sns.distplot(predictions,ax=ax2,bins=50)
ax2.set(title="predicted y value from test data of multiple x values")
# I can see similar distribution of data between two of them

# I will submit this code
# For that, I need to input predicted values into sampleSubmission.csv file
# First, I load that file
submission = pd.read_csv("D://chromedown//sampleSubmission.csv")
# I input predictions into submission's count column
submission["count"] = predictions
# print(submission.shape)
# (6493, 2)
print(submission.head())
#               datetime  count
# 0  2011-01-20 00:00:00  12.82
# 1  2011-01-20 01:00:00   5.12
# 2  2011-01-20 02:00:00   4.18
# 3  2011-01-20 03:00:00   3.56
# 4  2011-01-20 04:00:00   3.23

# I create a file storing score in the contents and file name
submission.to_csv("D://chromedown//Score_{0:.5f}_submission.csv".format(score), index=False)


plt.show()