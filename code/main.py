import pandas as pd

from param import parameter_parser
from MMGCN import MMGCN
from dataprocessing import data_pro
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt

def train(model, train_data, optimizer, epochs):
    model.train()
    for epoch in range(0, epochs):
        model.zero_grad() ## 梯度清零
        score = model(train_data) ## 得到预测值
        loss = torch.nn.MSELoss(reduction='mean')  ## 求解loss
        loss = loss(score, train_data['md_p'].to(device))
        loss.backward() ## 反向传播求解梯度
        optimizer.step() ## 更新权重参数
        #print("loss: ",loss.item())
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin) #Min-Max归一化方法
    return score


def main():
    from sklearn.metrics import roc_curve, auc
    args = parameter_parser()
    dataset = data_pro(args)
    train_data = dataset
    model = MMGCN(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    aucs = []
    aucs.append(0)
    epochs = []
    epochs.append(0)
    df2 = pd.read_csv("../datasets/m_d.csv")
    y = df2
    y = y.to_numpy().flatten()
    for i in range (1,700):
        scores = train(model, train_data, optimizer, i)
        df1 = pd.DataFrame(scores)
        y_pred = df1.iloc[0:-1,]
        y_pred = y_pred.to_numpy().flatten()
        fpr,tpr,threshold=roc_curve(y,y_pred)
        AUC = auc(fpr,tpr)
        aucs.append(AUC)
        epochs.append(i)
        print("auc: ",i,"=",AUC)

    print("aucs: ",aucs)
    plt.figure(figsize=(5,5),dpi=100)
    plt.plot(epochs,aucs,linestyle='-',label='gcn layer =3')

    plt.xlabel('Epoch')
    plt.ylabel('GCN layer')

    plt.legend()

    plt.show()

    #df = pd.DataFrame(score)
    #df.to_excel(r"C:\Users\zhr\Desktop\test.xlsx", sheet_name='Sheet1', index=False)
if __name__ == "__main__":
    main()
