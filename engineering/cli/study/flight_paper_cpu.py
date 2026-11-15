"""Darrah等人两段式网络的同输入CPU适配，模型接收一趟飞行的固定长度序列。"""
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn


class ConditionNetwork(nn.Sequential):
    def __init__(self):
        layers = []
        dims = (4,70,105,140,105,70,14)
        for i,(a,b) in enumerate(zip(dims[:-1],dims[1:])):
            layers.append(nn.Linear(a,b))
            if i+1 < len(dims)-1:
                layers.append(nn.ReLU())
        super().__init__(*layers)


class LifeNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.LSTM(14,32,num_layers=3,dropout=.5,bidirectional=True,batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.5),nn.Linear(64,1))

    def forward(self,x):
        _,(last,_) = self.layers(x)
        return self.head(torch.cat((last[-2],last[-1]),dim=1)).flatten()


def fit_epochs(model, x, y, valid_x, valid_y, epochs, batch, lr, decay=0., gamma=1.):
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma)
    best, best_epoch, saved = float("inf"), 0, None
    loss = nn.MSELoss()
    for epoch in range(1,epochs+1):
        model.train()
        for ix in torch.randperm(len(x)).split(batch):
            optimizer.zero_grad()
            loss(model(x[ix]),y[ix]).backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            error = loss(torch.cat([model(part) for part in valid_x.split(batch)]),valid_y).item()
        if error < best:
            best,best_epoch = error,epoch
            saved = {k:v.detach().clone() for k,v in model.state_dict().items()}
        if epoch-best_epoch >= 8:
            break
    if saved is None:
        raise ValueError("论文模型没有有限的确认结果")
    model.load_state_dict(saved)
    return best_epoch


def residual(x, model, stats):
    with torch.no_grad():
        z = (x-stats["mean"])/stats["scale"]
        return z[:,:,:14] - model(z[:,:,14:])


def train(path, destination):
    data = np.load(path)
    x = torch.as_tensor(data["x"])
    y = torch.as_tensor(data["y"],dtype=torch.float32)
    early, valid = (torch.as_tensor(data[k]) for k in ("early","valid"))
    train = ~valid
    if not (early&train).any() or not (early&valid).any():
        raise ValueError("源设备开工训练或确认样本不足")
    ref = x[train].reshape(-1,18)
    stats={"mean":ref.mean(0),"scale":ref.std(0).clamp_min(1e-6),
           "ymean":y[train].mean(),"yscale":y[train].std().clamp_min(1e-6)}
    z = (x-stats["mean"])/stats["scale"]
    train_h = z[train&early].reshape(-1,18)
    valid_h = z[valid&early].reshape(-1,18)
    healthy = ConditionNetwork()
    ep1=fit_epochs(healthy,train_h[:,14:],train_h[:,:14],valid_h[:,14:],valid_h[:,:14],25,256,.001,gamma=.95)
    r = residual(x,healthy,stats)
    target = (y-stats["ymean"])/stats["yscale"]
    life = LifeNetwork()
    ep2=fit_epochs(life,r[train],target[train],r[valid],target[valid],35,64,.0025,decay=.0001)
    torch.save({"healthy":healthy.state_dict(),"life":life.state_dict(),"stats":stats,
                "epochs":[ep1,ep2]},destination)
    print(f"健康参照发布第{ep1}轮，寿命模型发布第{ep2}轮。",flush=True)


def predict(inputs, outputs, artifact):
    bundle=torch.load(artifact,map_location="cpu",weights_only=True)
    healthy,life=ConditionNetwork(),LifeNetwork()
    healthy.load_state_dict(bundle["healthy"])
    life.load_state_dict(bundle["life"])
    healthy.eval()
    life.eval()
    stats=bundle["stats"]
    with torch.no_grad():
        rows=torch.as_tensor(np.load(inputs))
        values=torch.cat([life(residual(b,healthy,stats))*stats["yscale"]+stats["ymean"] for b in rows.split(64)])
    np.save(outputs,values.clamp_min(0).numpy())


if __name__=="__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    parser=ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    t=sub.add_parser("train")
    t.add_argument("source")
    t.add_argument("artifact")
    p=sub.add_parser("predict")
    p.add_argument("inputs")
    p.add_argument("outputs")
    p.add_argument("artifact")
    args=parser.parse_args()
    if args.action=="train":
        train(args.source,args.artifact)
    else:
        predict(args.inputs,args.outputs,args.artifact)
