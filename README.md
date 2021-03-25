#### Python Homework
Python实训课后作业  
金融客户分类

#### Environment
* python 3.7.2  
* sklearn  0.23.2  
* pandas  1.1.5
* numpy  1.19.4

####Code
各种分类方法及其准确率  
判断模型不只是准确率  
还包括召回率，F1分树等
####聚类
`k_means.py` -- K均值聚类  -- 0.85  
`svm.py` -- 支持向量机聚类  
`gmm.py` -- 高斯混合模型聚类  -- 0.70  
`DBSCAN.py` --  DBSCAN密度聚类  -- 0.40  

####其他分类
`KNN.py` -- K近邻  -- 0.89  
`logistic.py` -- 逻辑回归  -- 0.90 ± 1  
`decision_tree.py` -- 决策树  -- 0.90 ± 1  
`naive_bayes.py` -- 朴素贝叶斯 -- 0.86

###File
`bank-full.csv` - 数据文件  
`bank-names.txt` - 数据字段含义  
`tree.dot` - 决策树结构  
`tree.png` - 决策树效果图

####Process
* 导入文件观察数据情况 总数据 45000
* 本案例中unknown值较多  直接dropna只剩下 7800  
因此选择填充，当前采用频率最高词填充  
一些字段自己可用均值填充请自行完成
* 相关参数为分类例如本案例中的职业Job  
可转化为数字编码代表其含义 同时方便训练
* 根据数据情况选择合适的特征工程
* 使用模型训练
* 分析结果


####Remark
* 随便一个都可以分类  
所用算法可自行百度

* 决策树配合随机森林训练时间很长  
决策树可生成dot文件  
使用Graphivz生成图片

* svm训练时间很长  

* 效果最差的是DBSCAN！！！

* 仅供学习交流，严禁用于商业用途，请于24小时内删除