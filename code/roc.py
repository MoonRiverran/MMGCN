import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


df1 = pd.read_excel(r"C:\Users\zhr\Desktop\test.xlsx",index_col=None)
df2 = pd.read_csv("../datasets/m_d.csv")
y_pred = df1.iloc[0:-1,]
y_pred = y_pred.to_numpy().flatten()
y = df2
y = y.to_numpy().flatten()

fpr,tpr,threshold=roc_curve(y,y_pred)
auc = auc(fpr,tpr)

plt.figure(figsize=(5,5),dpi=100)
plt.plot(fpr,tpr,linestyle='-',label='MMGCN(auc=%0.3f)'%auc)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()


precision, recall, thresholds = precision_recall_curve(y,y_pred)
plt.plot(recall,precision,linestyle='-',label='MMGCN(auprc=%0.3f)'%auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

average_precision = average_precision_score(y,y_pred)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

plt.show()
