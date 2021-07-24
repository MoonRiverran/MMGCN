import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc

data = pd.read_csv("../datasets/diabetes.csv")
X = data.iloc[:,0:-1].values
y = data.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
model_SVC = SVC(kernel = 'rbf',random_state=4)
model_SVC.fit(X_train,y_train)

y_pred_svm = model_SVC.decision_function(X_test)
model_logistic = LogisticRegression()
model_logistic.fit(X_train,y_train)

y_pred_logistic = model_logistic.decision_function(X_test)

svm_fpr,svm_tpr,threshold = roc_curve(y_test,y_pred_svm)
auc_svm = auc(svm_fpr,svm_tpr)

logistic_fpr,logistic_tpr,threshold = roc_curve(y_test,y_pred_logistic)
auc_logistic = auc(logistic_fpr,logistic_tpr)

plt.figure(figsize=(5,5),dpi=100)
plt.plot(svm_fpr,svm_tpr,linestyle='-',label='SVM(auc=%0.3f)'%auc_svm)
plt.plot(logistic_fpr,logistic_tpr,marker='.',label='Logistic(auc=%0.3f)'%auc_logistic)

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()