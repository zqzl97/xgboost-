import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 使用交叉验证，这里没有用测试集数据
train_data = pd.read_csv('train.csv', index_col=0)
y_train = train_data['SalePrice']
x_train = train_data.drop(['SalePrice'], axis=1)

# 判断平滑性
# y_train.hist()
# 发现平滑性不怎么好，最好分布图类似于一个正太分布比较好
y_train_log = np.log1p(y_train)
# y_train_log.hist()

# MSSubClass是用数字表示，int型。作为类别标签看待，转为字符串
train_data['MSSubClass'] = train_data['MSSubClass'].astype(str)
# print(train_data['Condition1'].value_counts())

# one-hot
dummy_train = pd.get_dummies(train_data)
# print(len(dummy_train.columns))  303
# print(dummy_train.head())
# print(dummy_train.isnull().sum().sort_values(ascending=False).head(10))
mean_cols = dummy_train.mean()
# print(mean_cols)
dummy_train = dummy_train.fillna(mean_cols)
# 这304列 每一列里面有几个null
# print(dummy_train.isnull().sum())
# 304个0相加
# print(dummy_train.isnull().sum().sum())


# 标准化
# numeric_cols1 = train_data.columns[train_data.dtypes == 'object']
# print(numeric_cols1)
# 估计object的被one-hot了，！object的需要被标准化
numeric_cols = train_data.columns[train_data.dtypes != 'object']
# print(len(numeric_cols))
# print(numeric_cols)
# print(dummy_train.columns)
numeric_cols_means = dummy_train.loc[:, numeric_cols].mean()
numeric_cols_std = dummy_train.loc[:, numeric_cols].std()
# print(numeric_cols_means.shape) 36
# print(numeric_cols_std.shape) 36
# print(dummy_train.loc[:, numeric_cols].shape) (1460, 36)
dummy_train.loc[:, numeric_cols] = (dummy_train.loc[:, numeric_cols] - numeric_cols_means) / numeric_cols_std
# print(dummy_train)
# dummy_train.loc[:,numeric_cols] = (dummy_train.loc[:,numeric_cols]-numeric_col_means)/numeric_col_std

# 训练模型
# 通过交叉验证，看模型选用哪一套算法比较好
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

y_train = dummy_train['SalePrice']
X_train = dummy_train.drop(['SalePrice'], axis=1) # 丢掉这一列

alphas = np.logspace(-3, 2, 50)
test_scores = []
for alpha in alphas:
    clf = Ridge(alpha)  # 使用的是岭回归模型
    # 交叉验证
    # test_score = np.sqrt(-cross_val_score(clf, X_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    # 这里直接拿不平滑的y_train，即没有做log变化的SalePrice来做训练，均方误差在0.37左右。alpha = 20
    test_score = np.sqrt(-cross_val_score(clf, X_train, y_train_log, cv=10, scoring='neg_mean_squared_error'))
    # alpha = 15 均方误差约为0.1355，均方误差进一步减小了，说明平滑处理的重要性。
    test_scores.append(np.mean(test_score))
# 结果可视化
plt.plot(alphas, test_scores)
plt.title('Alpha vs CV Error')
plt.show()


# Adaboost
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score
params = [10, 15, 20, 25, 30, 35, 40]
test_scores = []
for param in params:
    clf = AdaBoostRegressor(n_estimators=param)
    # 默认情况下的弱分类器DecisionTreeRegressor
    test_score = np.sqrt(-cross_val_score(clf, dummy_train, y_train_log, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title('Adaboost_params vs CV Error')
plt.show()


# xgboost
from xgboost import XGBRegressor
params = [1, 2, 3, 4, 5, 6]
test_scores = []
for param in params:
    clf = XGBRegressor(max_depth=param, objective='reg:squarederror')
    test_score = np.sqrt(-cross_val_score(clf, dummy_train, y_train_log, cv=10, scoring='neg_mean_squared_error'))
    test_scores.append(np.mean(test_score))

plt.plot(params, test_scores)
plt.title('Xgboost_params vs CV Error')
plt.show()




