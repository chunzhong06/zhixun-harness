"""为数值回归流执行源码计划，共享同源码同训练输入的不可变权重。"""
from hashlib import sha256
from pathlib import Path
import shutil
from threading import RLock

import joblib
import numpy as np

from engineering.cli.workflow.london_code_driver import fit_source
from engineering.cli.workflow.london_midstream import identity, selected_rows, summarize
from engineering.cli.study.verify_generated_probe import predict
from harness.integrations.candidate_edit import named_scores
from harness.integrations.generated_regressor import load_module
from harness.models.sparse_reference import correct, setting_of
from harness.models.station_residual import error
from harness.runtime.storage import atomic_json


class StreamPlanBank:
    """训练工作者只收到合法拟合标签和过去输入；服务推理不改变已发布权重。"""
    def __init__(self, folder, features, names, reuse=None):
        self.folder=Path(folder);self.features=features;self.names=names
        self.cache={};self.errors={};self.predictions={};self.lock=RLock();self.fit_attempts=0;self.logical_slots=0
        self.reuse=Path(reuse) if reuse else None;self.reused_fits=0

    def component(self, source, selected, labels, history):
        x,y=self.features[selected],labels[selected]
        key=sha256(source.encode()+x.tobytes()+y.tobytes()).hexdigest()
        # 两个驱动可能同时提出等价组件，只执行一次实际拟合，费用不重复记账。
        with self.lock:
            self.logical_slots+=1
            if key in self.errors:raise ValueError(self.errors[key])
            if key in self.cache:return self.cache[key]
            folder=self.folder/key;folder.mkdir(parents=True,exist_ok=True)
            if self.reuse and (self.reuse/key/'identity.json').is_file():
                import json
                origin=self.reuse/key;record=json.loads((origin/'identity.json').read_bytes())
                with np.load(origin/'input.npz',allow_pickle=False) as z:
                    if not np.array_equal(z['train_x'],x) or not np.array_equal(z['train_y'],y) or not np.array_equal(z['train_ids'],np.flatnonzero(selected)):
                        raise ValueError('重用权重的训练数组不一致')
                if (origin/'fit/source.py').read_text(encoding='utf-8')!=source or sha256((origin/'fit/model.joblib').read_bytes()).hexdigest()!=record['weights_sha256']:
                    raise ValueError('重用源码或权重身份不同')
                shutil.copy2(origin/'input.npz',folder/'input.npz');shutil.copytree(origin/'fit',folder/'fit')
                record={**record,'folder':str(folder),'reused_from':str(origin)}
                atomic_json(folder/'identity.json',record);self.cache[key]=record;self.reused_fits+=1
                return record
            self.fit_attempts+=1
            data=folder/'input.npz'
            np.savez_compressed(data,train_x=x,train_y=y,train_ids=np.flatnonzero(selected),
                                predict_x=self.features[history],predict_ids=history)
            limit=1000.*max(1.,float(np.max(np.abs(y))))
            try:
                ids,values=fit_source(source,data,folder/'fit')
                if not np.array_equal(ids,history):raise ValueError('历史输入身份变化')
                with np.load(folder/'fit/base.npz',allow_pickle=False) as z:
                    if np.max(np.abs(z['raw_prediction']))>limit:raise ValueError('历史预测量级超限')
                record={'id':key,'source_sha256':sha256(source.encode()).hexdigest(),
                    'weights_sha256':sha256((folder/'fit/model.joblib').read_bytes()).hexdigest(),
                    'provided_training_rows':int(selected.sum()),'provided_last_training_hour':int(np.flatnonzero(selected)[-1]),
                    'prediction_limit':limit,'folder':str(folder)}
                atomic_json(folder/'identity.json',record);self.cache[key]=record
                return record
            except Exception as exc:
                self.errors[key]=str(exc);atomic_json(folder/'failed.json',{'error':str(exc)})
                raise

    def base(self, component, hours, fallback, *, strict=False):
        folder=Path(component['folder'])/'fit';hours=np.asarray(hours,dtype=int)
        key=(component['id'],sha256(hours.tobytes()).hexdigest())
        # 历史成功预测可复用；含逐行回退的预测取决于当前在用版，不能跨场景缓存。
        with self.lock:
            if key in self.predictions:return self.predictions[key].copy()
            weights=folder/'model.joblib'
            if sha256(weights.read_bytes()).hexdigest()!=component['weights_sha256']:raise ValueError('权重变化')
            load_module(folder/'source.py');model=joblib.load(weights)
            _,values,failures=predict(model,self.features[hours],component['prediction_limit'],fallback)
            if strict and failures:raise ValueError('历史重载出现无效预测')
            if not failures:self.predictions[key]=values.copy()
            else:atomic_json(Path(component['folder'])/('fallback-'+key[1]+'.json'),{'hours':hours.tolist(),'failures':failures})
            return values

    def train(self, plan, folder, features, names, fit, labels, history, known, windows):
        """兼容已有提案器；探索与确认只评分，绝不进入当前试训的拟合数组。"""
        if features is not self.features or names!=self.names:raise ValueError('执行器特征契约不同')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);atomic_json(folder/'plan.json',plan)
        selected=selected_rows(plan,fit,features,names);components=[];columns=[]
        for component in plan['components']:
            record=self.component(component['python_source'],selected,labels,history);components.append(record)
            values=self.base(record,history,np.zeros(len(history)),strict=True)
            columns.append(correct(values[:,None],known,history,setting_of(component['feedback']))[:,0])
        total=sum(c['weight']*p for c,p in zip(plan['components'],columns,strict=True))
        # 缺反馈窗口没有可比较分数，但固定配方仍能训练并直接交付。
        scores=[error(known[m[history]],total[m[history]]) if m[history].any() else None for m in windows]
        ablations=[]
        if len(columns)>1 and all(m[history].any() for m in windows[:2]):
            for i,component in enumerate(plan['components']):
                values=(total-component['weight']*columns[i])/(1-component['weight'])
                ablations.append({'removed_component':i,'remaining_weights_renormalized':True,
                    'scores':named_scores([error(known[m[history]],values[m[history]]) for m in windows[:2]],
                        windows[:2],history,prediction_object=f'ensemble_without_component_{i}')})
        record={'identity':identity(plan),'training_rows':int(selected.sum()),'last_training_hour':int(np.flatnonzero(selected)[-1]),
            'history_rmse':scores,'exploration':summarize(total,known,history,windows[:2],features,names),
            'components':components,'component_ablations':ablations,'prediction_limit':max(c['prediction_limit'] for c in components)}
        atomic_json(folder/'training.json',record);return record

    def issue(self, plan, training, history, service, labels, fallback):
        """先用该权重的合法历史初始化，再逐小时发放后接受本时刻到期参考。"""
        history=np.asarray(history,dtype=int);service=np.asarray(service,dtype=int);clock=np.r_[history,service]
        if len(history) and len(service) and history[-1]>=service[0]:raise ValueError('热身与服务时钟相交')
        total=np.zeros(len(service))
        for component,record in zip(plan['components'],training['components'],strict=True):
            past=self.base(record,history,np.zeros(len(history)),strict=True)
            future=self.base(record,service,fallback)
            values=correct(np.r_[past,future][:,None],labels[clock],clock,setting_of(component['feedback']))[:,0]
            total+=component['weight']*values[len(history):]
        return total
