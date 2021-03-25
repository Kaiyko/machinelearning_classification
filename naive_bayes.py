#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, mean_squared_error


def naive_bayes():
    """朴素贝叶斯"""

    #   需要进行标签装换成数字的列，缺失值严重的列
    # label_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    # loss_column = ['education', 'contact', 'poutcome']

    label_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    loss_column = ['education', 'contact', 'poutcome']

    #   读取数据，按分号 ； 隔开
    data = pd.read_csv("./bank-full.csv", sep=';')

    #   将unknown统一转化为 Nan
    data = data.replace(to_replace='unknown', value=np.nan)

    # data = data.fillna(method='ffill')
    # data = data.fillna(method='bfill')

    #   将 Nan值 即 unknown 转化为当前列频率最高的词
    # simple = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # for col in loss_column:
    #     data[col] = simple.fit_transform(data[col].values.reshape(-1, 1))

    #   缺失值移除
    data = data.dropna()

    #   将分类转化为数字
    label = LabelEncoder()
    for col in label_column:
        data[col] = label.fit_transform(data[col]).reshape(-1, 1)
        data[col] = label.fit_transform(data[col]).reshape(-1, 1)

    # del data['education']
    # del data['contact']
    # del data['poutcome']

    #   拆分训练集 测试集
    x_train, x_test, y_train, y_test = train_test_split(data[data.columns.values[0:16]],
                                                        data[data.columns.values[16]], test_size=0.25)

    #   数据归一化处理
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    print(x_train)

    #   朴素贝叶斯
    gau = GaussianNB()
    gau.fit(x_train, y_train)

    #   测试集分类
    y_predict = gau.predict(x_test)

    print("准确率: ", gau.score(x_test, y_test))
    print("均方误差: ", mean_squared_error(y_test, y_predict))
    print("召回率: \n", classification_report(y_test, y_predict, labels=[1, 0], target_names=["是", "否"]))


if __name__ == '__main__':
    naive_bayes()
