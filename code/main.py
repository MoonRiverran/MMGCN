import pandas as pd

from param import parameter_parser
from MMGCN import MMGCN
from dataprocessing import data_pro
import torch
import pandas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_data, optimizer, opt):
    model.train()
    for epoch in range(0, opt.epoch):
        model.zero_grad() ## 梯度清零
        score = model(train_data) ## 得到预测值
        loss = torch.nn.MSELoss(reduction='mean')  ## 求解loss
        loss = loss(score, train_data['md_p'].to(device))
        loss.backward() ## 反向传播求解梯度
        optimizer.step() ## 更新权重参数
        print("loss: ",loss.item())
    score = score.detach().cpu().numpy()
    scoremin, scoremax = score.min(), score.max()
    score = (score - scoremin) / (scoremax - scoremin) #Min-Max归一化方法
    return score


def main():
    args = parameter_parser()
    dataset = data_pro(args)
    train_data = dataset
    model = MMGCN(args)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    score = train(model, train_data, optimizer, args)
    df = pd.DataFrame(score)
    df.to_excel(r"C:\Users\23922\Desktop\test.xlsx", sheet_name='Sheet1', index=False)
if __name__ == "__main__":
    main()
