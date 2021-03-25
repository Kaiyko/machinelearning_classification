#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error


def k_means():
    """K-means"""

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

    #   结果不算入分类依据
    train = data[data.columns.values[0:16]]

    #   PCA降维 效果提升一丢丢
    pca = PCA(n_components=0.9)
    train = pca.fit_transform(train)

    #   K-Means聚类分析
    predict = KMeans(n_clusters=2).fit_predict(train)

    print("评估:     ", silhouette_score(train, predict))
    print("均方误差: ", mean_squared_error(data['y'], predict))
    print("召回率: \n", classification_report(data['y'], predict, labels=[1, 0], target_names=["是", "否"]))


if __name__ == '__main__':
    k_means()
