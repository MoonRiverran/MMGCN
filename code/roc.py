import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("../datasets/train.csv")
# 通过info查看数据信息
train.info()

# 对分类变量进行填充
train['Cabin'] = train['Cabin'].fillna('NA')
train['Embarked'] = train['Embarked'].fillna('S')

# 对连续变量进行填充
train['Age'] = train['Age'].fillna(train['Age'].mean())

train.isnull().mean().sort_values(ascending=False)
data = train[['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare', 'Embarked']]
data = pd.get_dummies(data)
data.head()

# 提出Survived作为数据集标签y
y = train['Survived']
# 使用sklearn拆分数据集为训练集和测试集
from sklearn.model_selection import train_test_split
# stratify参数是为了保证分布均匀，random_state是为了复现结果，使得每次拆分结果一样
X_train, X_test, y_train, y_test = train_test_split(data, y, stratify=y, random_state=0)
# 查看一下拆分后的数据集大小
print(X_train.shape, X_test.shape)



# 默认参数逻辑回归模型
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)
# # 查看训练集和测试集分数
# train_score = lr.score(X_train, y_train)
# test_score = lr.score(X_test, y_test)
# print(f"train score: {train_score:.2f}")
# print(f"test score: {test_score:.2f}")
#
# # 随机森林分类模型
# rf = RandomForestClassifier()
# rf.fit(X_train, y_train)
# # 查看训练集和测试集分数
# train_score = rf.score(X_train, y_train)
# test_score = rf.score(X_test, y_test)
# print(f"train score: {train_score:.2f}")
# print(f"test score: {test_score:.2f}")
#
from sklearn.model_selection import cross_val_score
lr = LogisticRegression(max_iter=10000)
score = []
alphas = []
for alpha in range(1,100,1):
    alphas.append(alpha)
    sc = np.sqrt(cross_val_score(lr, X_train, y_train, cv=10))
    score.append(sc.mean())
plt.plot(alphas,score)
plt.show()
# # 交叉验证所有分数
# print(scores)
# # 交叉验证平均分
# print(f"Average cross-validation score: {scores.mean():.3f}")


#
# from sklearn.metrics import confusion_matrix
# # 训练模型
# lr = LogisticRegression(C=100)
# lr.fit(X_train, y_train)
# # 模型预测结果
# pred = lr.predict(X_train)
# # 混淆矩阵
# print(confusion_matrix(y_train, pred))
#
# from sklearn.metrics import classification_report
# # 精确率、召回率以及f1-score
# print(classification_report(y_train, pred))
#
# from sklearn.metrics import roc_curve
# fpr, tpr, thresholds = roc_curve(y_test, lr.decision_function(X_test))
# plt.plot(fpr, tpr, label="ROC Curve")
# plt.xlabel("FPR")
# plt.ylabel("TPR (recall)")
# # 找到最接近于0的阈值
# close_zero = np.argmin(np.abs(thresholds))
# plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10, label="threshold zero", fillstyle="none", c='k', mew=2)
# plt.legend(loc=4)
# plt.show()
