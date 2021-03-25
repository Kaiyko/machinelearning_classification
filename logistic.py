#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_squared_error, recall_score, confusion_matrix


def logistic():
    """逻辑回归 - sigmoid, softmax"""

    #   需要进行标签装换成数字的列，缺失值严重的列
    label_column = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
    loss_column = ['education', 'contact', 'poutcome', 'job']

    #   读取数据，按分号 ； 隔开
    data = pd.read_csv("./bank-full.csv", sep=';')

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

    #   拆分训练集 测试集
    x_train_under_sample, x_test_under_sample, y_train_under_sample, y_test_under_sample = train_test_split(x_under_sample, y_under_sample, test_size=0.25)

    #   交叉训练集 获得best c值
    best_c = kflod_scores(x_train_under_sample, y_train_under_sample)

    #   数据归一化处理
    std = StandardScaler()
    x_train_under_sample = std.fit_transform(x_train_under_sample)
    x_test_under_sample = std.transform(x_test_under_sample)

    #   逻辑回归
    lg = LogisticRegression(C=best_c, penalty='l1', solver='liblinear')
    lg.fit(x_train_under_sample, y_train_under_sample.values.ravel())
    #   测试集分类
    y_predict_under_sample = lg.predict(x_test_under_sample)

    print("准确率: ", lg.score(x_test_under_sample, y_test_under_sample))
    print("均方误差: ", mean_squared_error(y_test_under_sample, y_predict_under_sample))
    print("召回率: \n", classification_report(y_test_under_sample, y_predict_under_sample, labels=[1, 0], target_names=["是", "否"]))


def kflod_scores(x_train_data, y_train_data):
    """求出最佳的c值"""

    fold = KFold(5, shuffle=False)
    # fold.get_n_splits()
    c_param_range = [0.01, 0.1, 1, 10, 1000]

    results_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_parameter', 'Mean recall score'])
    results_table['C_parameter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        recall_accs = []
        for iteration, indices in enumerate(fold.split(x_train_data), start=1):
            lg = LogisticRegression(C=c_param, penalty='l1', solver='liblinear')

            lg.fit(x_train_data.iloc[indices[0], :], y_train_data.iloc[indices[0], :].values.ravel())

            y_pred_under_sample = lg.predict(x_train_data.iloc[indices[1], :].values)

            recall_acc = recall_score(y_train_data.iloc[indices[1], :].values, y_pred_under_sample)
            recall_accs.append(recall_acc)
            print('iteration', iteration, ': recall score = ', recall_acc)

        results_table.loc[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score ', np.mean(recall_accs))
        print('')

    best_c = results_table.loc[results_table['Mean recall score'].astype('float64').idxmax()]['C_parameter']
    print('*********************************************************************************')
    print('Best model to choose from cross validation is with C parameter = ', best_c)
    print('*********************************************************************************')
    return best_c


if __name__ == '__main__':
    logistic()
