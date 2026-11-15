"""离线作者TabICLv2候选：只读源标签和无标签预测输入，保存重载后逐行交付。"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

from argparse import ArgumentParser
from hashlib import sha256
import gc
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'engineering/.environments/tabicl-runtime'),str(ROOT/'engineering/src')]
import numpy as np
import pandas as pd
import torch
from tabicl import TabICLClassifier,TabICLRegressor
from harness.runtime.storage import atomic_json


def run(folder):
    torch.set_num_threads(1)
    spec=json.loads((folder/'spec.json').read_bytes());checkpoint=Path(spec['checkpoint'])
    if sha256(checkpoint.read_bytes()).hexdigest()!=spec['checkpoint_sha256']:raise ValueError('作者预训练权重改变')
    model_type=TabICLClassifier if spec['classification'] else TabICLRegressor
    model=model_type(n_estimators=8,batch_size=1,kv_cache=True,device='cpu',n_jobs=1,
                     model_path=checkpoint,allow_auto_download=False,random_state=42,use_amp=False)
    train=pd.read_csv(folder/'train.csv');y=train.pop('target').to_numpy()
    started=time.monotonic();model.fit(train,y);fit_seconds=time.monotonic()-started
    model.save(folder/'model.pkl',save_model_weights=False,save_training_data=True,save_kv_cache=True)
    del model;gc.collect();model=model_type.load(folder/'model.pkl',device='cpu')
    predictions={};started=time.monotonic()
    for phase in ['explore','confirm','future']:
        frame=pd.read_csv(folder/(phase+'.csv'))
        predictions[phase]=np.maximum(0.,[float(model.predict(frame.iloc[[i]])[0]) for i in range(len(frame))])
        if not np.isfinite(predictions[phase]).all():raise ValueError('实际预测非法')
        if spec['classification'] and not np.isin(predictions[phase],np.unique(y)).all():raise ValueError('实际类别非法')
        if phase=='explore':
            batch=np.maximum(0.,model.predict(frame))
            if not np.allclose(batch,predictions[phase],rtol=1e-5,atol=1e-4):raise ValueError('逐行和批量结果不一致')
    np.savez_compressed(folder/'predictions.npz',**predictions)
    record={'success':True,'fit_seconds':fit_seconds,'inference_seconds':time.monotonic()-started,
            'state_bytes':(folder/'model.pkl').stat().st_size,'state_sha256':sha256((folder/'model.pkl').read_bytes()).hexdigest(),
            'checkpoint_sha256':spec['checkpoint_sha256'],'predictions_sha256':sha256((folder/'predictions.npz').read_bytes()).hexdigest(),
            'source_rows':len(y),'features':train.shape[1],'single_row_batch_consistency':True,'n_estimators':8}
    atomic_json(folder/'complete.json',record);print(json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--folder',type=Path,required=True);run(p.parse_args().folder)
