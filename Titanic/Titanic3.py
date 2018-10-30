# -*- coding: utf-8 -*-

import numpy
import pandas
from sklearn.preprocessing import StandardScaler,scale
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import linear_model


# 使用随机森林回归 RandomForestRegressor 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即Age列
    y = known_age[:, 0]

    # X即其他特征属性列
    x = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(x, y)

    # 用得到的模型进行未知年龄结果预测
    predictedages = rfr.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedages

    return df, rfr


def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def set_test_missing_ages(df, rfr):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 未知年龄
    unknown_age = age_df[age_df.Age.isnull()].values
    predictedages = rfr.predict(unknown_age[:, 1:])

    df.loc[(df.Age.isnull()), 'Age'] = predictedages

    return df


if __name__ == "__main__":
    train = pandas.read_csv("all/train.csv")

    # 数据缺失值处理
    train, rfr = set_missing_ages(train)
    train = set_cabin_type(train)

    dummies_Cabin = pandas.get_dummies(train['Cabin'], prefix='Cabin')
    dummies_Embarked = pandas.get_dummies(train['Embarked'], prefix='Embarked')
    dummies_Sex = pandas.get_dummies(train['Sex'], prefix='Sex')
    dummies_Pclass = pandas.get_dummies(train['Pclass'], prefix='Pclass')
    df = pandas.concat([train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # Age和Fare特征化到[-1,1]之内
    #scaler = StandardScaler()
    #df['Age_scaled'] = scaler.fit_transform(df[['Age']])
    #df['Fare_scaled'] = scaler.fit_transform(df[['Fare']])

    # 训练出模型
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values

    # y即Survived结果
    y = train_np[:, 0]

    # X即特征属性值
    X = train_np[:, 1:]

    # 得到模型
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)

    # 测试数据也要预处理
    test = pandas.read_csv("all/test.csv")
    test2 = test

    test = set_test_missing_ages(test, rfr)
    test = set_cabin_type(test)

    dummies_Cabin = pandas.get_dummies(test['Cabin'], prefix='Cabin')
    dummies_Embarked = pandas.get_dummies(test['Embarked'], prefix='Embarked')
    dummies_Sex = pandas.get_dummies(test['Sex'], prefix='Sex')
    dummies_Pclass = pandas.get_dummies(test['Pclass'], prefix='Pclass')

    df_test = pandas.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    #df_test['Age_scaled'] = scaler.fit_transform(df_test[['Age']])
    #df_test['Fare_scaled'] = scaler.fit_transform(df_test[['Fare']])

    # 进行预测
    test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(test)
    submission = pandas.DataFrame({
        "PassengerId": test2["PassengerId"].values,
        "Survived": predictions.astype(numpy.int32)
    })

    # 保存结果
    submission.to_csv("all/DecisionTree_result3.csv", index=False)

    print("---finish---")
