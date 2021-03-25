#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def decision_tree_main():
    """决策树 + 随机森林"""

    #   需要进行标签装换成数字的列，缺失值严重的列
    label_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    loss_column = ['education', 'contact', 'poutcome', 'job']

    #   读取数据，按分号 ； 隔开
    data = pd.read_csv("./bank-full.csv", sep=';')
    print(data)

    #   将unknown统一转化为 Nan
    data = data.replace(to_replace='unknown', value=np.nan)
    print(data)

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
    print(data)

    #   过采样处理
    number_records_fraud = len(data[data.y == 1])
    fraud_indices = np.array(data[data.y == 1].index)

    normal_indices = data[data.y == 0].index

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    under_sample_data = data.iloc[under_sample_indices, :]

    x_under_sample = under_sample_data.iloc[:, under_sample_data.columns != 'y']
    y_under_sample = under_sample_data.iloc[:, under_sample_data.columns == 'y']

    print('normal:', len(under_sample_data[under_sample_data.y == 1]))
    print('fraud:', len(under_sample_data[under_sample_data.y == 0]))
    print('total', len(under_sample_data))

    #   直接拆分训练集 测试集
    # x_train, x_test, y_train, y_test = train_test_split(data[data.columns.values[0:16]],
    #                                                     data[data.columns.values[16]], test_size=0.25)

    #   过采样处理后拆分训练集 测试集
    x_train, x_test, y_train, y_test = train_test_split(x_under_sample,  y_under_sample, test_size=0.25)

    #   特征工程
    dict_vec = DictVectorizer(sparse=False)
    x_train = dict_vec.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict_vec.transform(x_test.to_dict(orient="records"))

    #   决策树训练
    decision_tree = DecisionTreeClassifier(max_depth=8)
    decision_tree.fit(x_train, y_train)
    y_predict = decision_tree.predict(x_test)

    print("准确率为: ", decision_tree.score(x_test, y_test))
    print("召回率: \n", classification_report(y_test, y_predict, labels=[1, 0], target_names=["是", "否"]))

    #   导出决策树 导出决策树为dot文件 后续自行用Graphviz生成png
    export_graphviz(decision_tree, out_file='./tree.dot',
                    feature_names=dict_vec.get_feature_names())

    #   随机森林
    random_forest = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500],
             "max_depth": [5, 8, 15, 25, 30]}
    grid_search_cv = GridSearchCV(random_forest, param_grid=param)
    grid_search_cv.fit(x_train, y_train)

    print("准确率：", grid_search_cv.score(x_test, y_test))
    print("查看选择的参数", grid_search_cv.best_params_)
    y_predict = grid_search_cv.predict(x_test)

    print("准确率为: ", grid_search_cv.score(x_test, y_test))
    print("召回率: \n", classification_report(y_test, y_predict, labels=[1, 0], target_names=["是", "否"]))


if __name__ == '__main__':
    decision_tree_main()
