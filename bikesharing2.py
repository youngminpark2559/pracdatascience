import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# 그래프에서 격자로 숫자 범위가 눈에 잘 띄도록 ggplot 스타일을 사용
plt.style.use('ggplot')

# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

# I load data and test file
train = pd.read_csv("D://chromedown//train.csv", parse_dates=["datetime"])
# print(train.shape)
# (10886, 12)
test = pd.read_csv("D://chromedown//test.csv", parse_dates=["datetime"])
# print(test.shape)
# (6493, 9)


# I put only year, month, hour, dayofweek data which will be used into train dataframe
# Notice that columns are increased
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
# print(train.shape)
# (10886, 16)
test["year"] = test["datetime"].dt.year
test["month"] = test["datetime"].dt.month
test["hour"] = test["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
# print(test.shape)
# (6493, 13)


# continuous type feature와 categorical type feature

# In the case of categorical feature, if you convert them into one hot encoding, you can get higher score


categorical_feature_names = ["season", "holiday", "workingday", "weather", "dayofweek", "month","year", "hour"]
# I convert categorical type feature to category data type
for var in categorical_feature_names:
    train[var] = train[var].astype("category")
    test[var] = test[var].astype("category")

# I choose following features
feature_names = ["season", "weather", "temp", "atemp", "humidity", "year", "hour", "dayofweek", "holiday", "workingday"]
# print(feature_names)
# ['season',
#  'weather',
#  'temp',
#  'atemp',
#  'humidity',
#  'year',
#  'hour',
#  'dayofweek',
#  'holiday',
#  'workingday']

# I bring chosen features from train data and put them into X_train
X_train = train[feature_names]
# print(X_train.shape)
# (10886, 10)
# X_train.head()

X_test = test[feature_names]
# print(X_test.shape)
# (6493, 10)
# X_test.head()

label_name = "count"
y_train = train[label_name]
# print(y_train.shape)
# (10886,)
# y_train.head()
# Out[11]:
# 0    16
# 1    40
# 2    32
# 3    13
# 4     1
# Name: count, dtype: int64


from sklearn.metrics import make_scorer
# I implement evaluation method
def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)
        
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    mean_difference = difference.mean()
    
    score = np.sqrt(mean_difference)
    
    return score


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

# I get linear regression model
lModel = LinearRegression()

# I make the linear regression model to learn by giving y_train(y_train_log), X_train
y_train_log = np.log1p(y_train)
lModel.fit(X_train, y_train_log)

# I input X_train data into linear regression model and get predicted value related to that X_train data
preds = lModel.predict(X_train)
# I input predicted data into rmsle to evaluate how it is precise
print("RMSLE Value For Linear Regression implemented above : ", rmsle(np.exp(y_train_log),np.exp(preds), False))


# One of regularization methods - Ridge
# 가중치(w)의 모든 원소가 0에 가깝게 만들어 모든 피처가 주는 영향을 최소화(기울기를 작게 만듦)
# Regularization is used to make purposed noise to the hypothesis function not to make the model have overfitting
ridge_m_ = Ridge()
# I try max_iter = 3000 in this test
# I will find optimal alpha value in 3000 iterations
ridge_params_ = { 'max_iter':[3000],'alpha':[0.01, 0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
# I execute the GridSearchCV() with passing arguments(ridge regularization, alpha, scorer, cv)
grid_ridge_m = GridSearchCV(ridge_m_,
                          ridge_params_,
                          scoring = rmsle_scorer,
                          cv=5)

y_train_log = np.log1p(y_train)
# I make the model to learn by passing train data and label
grid_ridge_m.fit(X_train, y_train_log)
# I make the model to expect values correspoding X_train
preds = grid_ridge_m.predict(X_train)
# I show best_params_
print(grid_ridge_m.best_params_)
print("RMSLE Value after Ridge Regularization : ", rmsle(np.exp(y_train_log),np.exp(preds), False))

# I visualize alpha and rmsle score
fig,ax = plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)

# {'alpha': 0.01, 'max_iter': 3000}
# RMSLE Value For Ridge Regression:  0.980369790278

# One of regularization methods - Lasso
# Lasso regularization which is call L1 regularization tries to make a coefficient closer to 0
# Some coefficient even becomes 0, which means the feature excluded from data can be occurred
# It also can be meant that automatic feature selection is performed by lasso regularization
# Default value of alpha is 1.0, to make underfitting decreased, I should make alpha value lower
# When I execute the following Lasso-regularized model by GridSearchCV(), best alpha value turns out to be 0.00125

lasso_m_ = Lasso()
alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

grid_lasso_m = GridSearchCV(lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)
y_train_log = np.log1p(y_train)
grid_lasso_m.fit( X_train , y_train_log )
preds = grid_lasso_m.predict(X_train)
print(grid_lasso_m.best_params_)
print("RMSLE Value For Lasso Regression: ",rmsle(np.exp(y_train_log),np.exp(preds),False))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
# {'alpha': 0.00125, 'max_iter': 3000}
# RMSLE Value For Lasso Regression:  0.980372782146


# Ensemble Models - component 1 : Random Forest
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=100)

# I preprocess y_train
y_train_log = np.log1p(y_train)
# I make the random Forest model to learn
rfModel.fit(X_train, y_train_log)

# I make the model to expect
preds = rfModel.predict(X_train)
score = rmsle(np.exp(y_train_log),np.exp(preds),False)
print ("RMSLE Value For Random Forest: ",score)
# RMSLE Value For Random Forest:  0.107075374495

# Ensemble Model - component 2 : Gradient Boost
# This is another ensemble technique by binding several dicision trees to make powerful expecting model
# It can be used for both regression and classification
# Unlike random forest, gradient boost makes tree in order as the way of complementing the error of tree
# You won't get randomness and prior pruning is used in this process
# Since not that deep tree of 1-5 steps is used, it consumes low memory, and has rapid expectation
# learning_rate : controls how it revises error
# If you increase n_estimators, you will get much more trees in your ensemble so that the complexity of model goes high
# You will get more chances to fix error in the training data but at the same time, it can lead to overfitting with more complex model
# max_depth(max_leaf_nodes) 

from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000, alpha=0.01);

y_train_log = np.log1p(y_train)
gbm.fit(X_train, y_train_log)

preds = gbm.predict(X_train)

score = rmsle(np.exp(y_train_log),np.exp(preds),False)

print ("RMSLE Value For Gradient Boost: ", score)
# RMSLE Value For Gradient Boost:  0.213574037272

predsTest = gbm.predict(X_test)
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
sns.distplot(np.exp(predsTest),ax=ax2,bins=50)

# Submit
submission = pd.read_csv("D://chromedown//sampleSubmission.csv")
# print(submission)

submission["count"] = np.exp(predsTest)

print(submission.shape)
print(submission.head())  
# (6493, 2)

submission.to_csv("D://chromedown//Score_{0:.5f}_submission.csv".format(score), index=False)

plt.show()