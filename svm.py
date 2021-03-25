#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV


def svm():
    """svm支持向量机 - 经典!"""

    #   所有列，需要进行标签装换成数字的列，缺失值严重的列
    label_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    loss_column = ['education', 'contact', 'poutcome']

    #   读取数据，按分号 ； 隔开
    data = pd.read_csv("./bank-full.csv", sep=r';')

    #   将unknown统一转化为 Nan
    data = data.replace(to_replace='unknown', value=np.nan)

    #   将 Nan值 即 unknown 转化为当前列频率最高的词
    simple = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    for col in loss_column:
        data[col] = simple.fit_transform(data[col].values.reshape(-1, 1))

    #   缺失值移除
    data = data.dropna()

    #   将分类转化为数字
    label = LabelEncoder()
    for col in label_column:
        data[col] = label.fit_transform(data[col]).reshape(-1, 1)
        data[col] = label.fit_transform(data[col]).reshape(-1, 1)

    #   拆分训练集 测试集
    x_train, x_test, y_train, y_test = train_test_split(data[data.columns.values[0:16]],
                                                        data[data.columns.values[16]], test_size=0.25)

    #   PCA降维 + SVC分类
    pca = PCA(n_components=0.9)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    #   选出最优参数组
    param_grid = {'svc__C': [1, 5, 10],
                  'svc__gamma': [0.0001, 0.0005, 0.001]}

    grid = GridSearchCV(model, param_grid)

    #   开始训练 时间很长！
    grid.fit(x_train, y_train)
    print(grid.best_params_)

    #   用最优参数组进行分类
    model = grid.best_estimator_
    predict = model.predict(x_test)

    print("准确率: ", model.score(x_test, y_test))
    print("均方误差: ", mean_squared_error(y_test, predict))
    print("召回率: \n", classification_report(y_test, predict, labels=[1, 0], target_names=["是", "否"]))


if __name__ == '__main__':
    svm()
