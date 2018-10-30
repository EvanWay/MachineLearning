# -*- coding: utf-8 -*-

import pandas
from sklearn.tree import DecisionTreeClassifier

print("加载数据")
train = pandas.read_csv("all/train.csv")
test = pandas.read_csv("all/test.csv")
print(train.info(), test.info())

# 处理Age，取中位数
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# 处理Sex
train["Sex"] = train["Sex"].apply(lambda x: 1 if x == "male" else 0)
test["Sex"] = test["Sex"].apply(lambda x: 1 if x == "male" else 0)

# 特征选择
feature = ["Age", "Sex"]

# 模型的选择：sciket-learn提供的决策树
dt = DecisionTreeClassifier()
dt = dt.fit(train[feature], train["Survived"])

# 进行预测
predict_data = dt.predict(test[feature])
submission = pandas.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": predict_data
})

# 保存结果
submission.to_csv("all/DecisionTree_result.csv", index=False)

print("---finish---")
