"""冻结TabPFN教师与原父分类网的算术组合；原源修正在组合之后执行。"""
from hashlib import sha256
import importlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

if __package__:
    from .tabular_classifier import TabICLClassifierModel,classification_base
    from .frozen_attention import _enable_frozen_cpu_inference
    from .probability_mixture import arithmetic_output
    from .quantized_teacher import restore_teacher
    from .teacher_cache import frozen_batch_values, frozen_teacher_values
    try:
        from .complete_checkpoint import FORMAT
    except ImportError:
        from .complete_model import FORMAT
else:
    from tabular_classifier import TabICLClassifierModel,classification_base
    from frozen_attention import _enable_frozen_cpu_inference
    from probability_mixture import arithmetic_output
    from quantized_teacher import restore_teacher
    from teacher_cache import frozen_batch_values, frozen_teacher_values
    from complete_model import FORMAT

KIND='tabpfn-classification-parent'
BASELINE_ENSEMBLES = {'gas-baseline-6plus2-v1': (6, 2), 'gas-baseline-3plus1-v1': (3, 1)}


def baseline_context_available(x, y):
    return (x.ndim==2 and x.shape==(len(y),128) and y.ndim==1
            and np.array_equal(np.unique(y),np.arange(6))
            and sum(np.count_nonzero(y==i)//2 for i in range(6))>=2)


def baseline_context(x,y):
    """只在当前源训练内为半数行换本底，使用整组传感器并保留相对反应。"""
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y)
    if not baseline_context_available(x,y) or not np.isfinite(x).all():
        raise ValueError('训练资料无法构建六类气体的本底上下文')
    rng=np.random.default_rng(42)
    selected=[]
    for label in range(6):
        rows=rng.permutation(np.flatnonzero(y==label))
        selected.extend(rows[:len(rows)//2])
    order=rng.permutation(selected)
    donor=np.arange(len(x))
    donor[order]=np.roll(order,1)
    sensors=x.reshape(-1,16,8)
    diff,den=sensors[:,:,0],sensors[:,:,1]-1
    baseline=np.divide(diff,den,out=np.full_like(diff,np.nan),where=np.abs(den)>1e-5)
    valid=np.isfinite(baseline)&(baseline>0)
    ratio=np.divide(baseline[donor],baseline,out=np.ones_like(baseline),where=valid&valid[donor])
    transformed=sensors.copy()
    transformed[:,:,0]*=ratio
    transformed[:,:,2:]*=ratio[:,:,None]
    result=transformed.reshape(x.shape).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('基线交换后的源幅值超出单精度范围')
    return result


class _FrozenClassifier:
    """仅在已拟合教师的预测调用内启用后端；不同成员保有独立的源上下文。"""

    def __init__(self, predictor, backend):
        self.predictor = predictor
        self.backend = backend

    def predict_proba(self, rows):
        token = self.backend.active.set(True)
        try:
            return self.predictor.predict_proba(rows)
        finally:
            self.backend.active.reset(token)


def _teacher_threads():
    requested = os.environ.get('ZHIXUN_SDK_PREDICT_THREADS')
    return int(requested) if requested in ('2', '4') else 1


def _prediction_teacher(predictor):
    if os.environ.get('ZHIXUN_SDK_PREDICT_THREADS') in ('1', '2', '4'):
        _enable_frozen_cpu_inference(predictor)
        backend = getattr(predictor, '_zhixun_cpu_backend', None)
        if backend is not None:
            return _FrozenClassifier(predictor, backend)
    return predictor


class _BaselineEnsemble:
    """共享一份发布资产；六个原始成员和两个本底变换成员按成员数投票。"""
    def __init__(self,raw,alternate):
        self.raw,self.alternate=raw,alternate

    def predict_proba(self,raw):
        return .75*self.raw.predict_proba(raw)+.25*self.alternate.predict_proba(raw)


def author_package():
    """优先随发布的固定作者库；研发回退只使用项目既有隔离运行库。"""
    bundled=Path(__file__).resolve().parent.parent
    if (bundled/'tabpfn/__init__.py').is_file():sys.path.insert(0,str(bundled))
    try:
        return importlib.import_module('tabpfn')
    except ModuleNotFoundError as exc:
        if exc.name!='tabpfn' or not (__package__ or '').startswith('harness.'):raise
        from harness.paths import workspace_root
        root=workspace_root()
        runtime=root/'engineering/.environments/tabpfn35-runtime'
        if not (runtime/'tabpfn/__init__.py').is_file():raise RuntimeError('缺少固定TabPFN运行库') from exc
        sys.path[:0]=[str(runtime),str(root/'engineering/.environments/tabicl-runtime')]
        return importlib.import_module('tabpfn')


def prediction_dependencies():
    """发布当前平台的作者库与必要依赖，排除测试和缓存；Linux另在镜像内打包。"""
    files={};previous=os.environ.get('SKB_DATA_DIRECTORY')
    try:
        with TemporaryDirectory(prefix='harness-tabpfn-pack-') as temporary:
            os.environ['SKB_DATA_DIRECTORY']=temporary
            author_package()
            for dependency in ('tabpfn','skrub','safetensors','huggingface_hub','einops','tqdm'):
                module=importlib.import_module(dependency);folder=Path(module.__file__).parent
                for path in folder.rglob('*'):
                    relative=path.relative_to(folder)
                    if not path.is_file() or set(relative.parts)&{'__pycache__','tests','_tests','docs','benchmarks'}:continue
                    if path.suffix in ('.py','.pyi','.json','.so','.pyd','.txt') or path.name=='py.typed':
                        files[dependency+'/'+relative.as_posix()]=path.read_bytes()
                metadata=list(folder.parent.glob(dependency.replace('-','_')+'-*.dist-info'))
                if len(metadata)!=1:raise ValueError('固定依赖版本来源不唯一：'+dependency)
                for path in metadata[0].rglob('*'):
                    if path.is_file() and (path.name in ('METADATA','WHEEL','LICENSE','LICENSE.txt') or 'licenses' in path.parts):
                        files[path.relative_to(folder.parent).as_posix()]=path.read_bytes()
    finally:
        if previous is None:os.environ.pop('SKB_DATA_DIRECTORY',None)
        else:os.environ['SKB_DATA_DIRECTORY']=previous
    return files


class TabPFNClassifierModel(TabICLClassifierModel):
    """沿用源行缓存及原修正执行，单独定义作者加载、资产检查和算术梯度。"""
    def __init__(self,base,auxiliary_arrays):
        self.base=base;self.auxiliary_arrays={k:v.copy() for k,v in auxiliary_arrays.items()}
        self.state=json.loads(str(self.auxiliary_arrays['complete_state']))
        self.teacher_state=json.loads(str(self.auxiliary_arrays['tabpfn_classifier_state']))
        if self.state.get('format')!=FORMAT or self.state.get('kind')!=KIND:
            raise ValueError('未知TabPFN分类组合格式')
        if self.teacher_state.get('format')!='harness.tabpfn-classifier-resource.v1':
            raise ValueError('未知TabPFN教师资源格式')
        x,y=self.auxiliary_arrays['classifier_source_x'],self.auxiliary_arrays['classifier_source_y']
        if x.ndim!=2 or x.shape[1]!=len(base.meta['paths']) or y.shape!=(len(x),):
            raise ValueError('分类源上下文与实际父输入不匹配')
        if not np.isfinite(x).all() or set(y)!=set(range(len(base.meta['classes']))):
            raise ValueError('分类源上下文需覆盖完整类别且输入有限')
        payload=self.auxiliary_arrays['teacher_asset']
        if payload.dtype!=np.uint8 or payload.ndim!=1 or sha256(payload.tobytes()).hexdigest()!=self.teacher_state['asset_sha256']:
            raise ValueError('冻结教师压缩资产身份不符')
        self._check_context()
        self._check_drift_readout()
        if 'toluene_readout' in self.state or any(k.startswith('gas_svm_') for k in self.auxiliary_arrays):
            if __package__:
                from .gas_toluene_readout import validate
            else:
                from gas_toluene_readout import validate
            validate(self.state.get('toluene_readout'),self.auxiliary_arrays,base.meta['classes'])
        if 'kinetic_readout' in self.state or any(k.startswith('kinetic_support_') for k in self.auxiliary_arrays):
            if __package__:
                from .gas_kinetic_support import validate as validate_kinetics
            else:
                from gas_kinetic_support import validate as validate_kinetics
            validate_kinetics(self.state.get('kinetic_readout'),self.auxiliary_arrays,base.meta['classes'])
        self._teacher=None;self._training_teacher={};self._prediction_cache={};self._repair=None
        self._round_cache = None
        self.round_cache_hits = {'training':0,'evaluation':0}
        self.reset_device_context()
        if 'device_context' in self.state or any(k.startswith('device_context_') for k in self.auxiliary_arrays):
            self._device_module().validate(self.state.get('device_context'), self.auxiliary_arrays, base.meta['classes'])
        self.validate()

    @staticmethod
    def _device_module():
        if __package__:
            from . import gas_device_context
        else:
            import gas_device_context
        return gas_device_context

    def reset_device_context(self):
        """当轮接受新上下文时丢弃旧拟合缓存，下一次实际命中时才加载额外的教师。"""
        self._device_teacher = None
        self._device_backgrounds = None
        self._device_batch_outputs = {}

    def _device_probabilities(self, matched):
        arrays = self.auxiliary_arrays
        if self._device_teacher is None:
            import torch
            torch.set_num_threads(_teacher_threads())
            torch.manual_seed(42)
            with TemporaryDirectory(prefix='harness-device-tabpfn-') as temporary:
                os.environ['SKB_DATA_DIRECTORY'] = temporary
                path = Path(temporary)/'tabpfn-v3.5-fast-20260909.safetensors'
                restore_teacher(arrays['teacher_asset'], path)
                estimator = author_package().TabPFNClassifier(model_path=path,n_estimators=self.teacher_state['n_estimators'],device='cpu',random_state=42,
                    fit_mode='fit_with_cache',n_preprocessing_jobs=1,inference_precision=torch.float32)
                estimator.fit(arrays['device_context_x'].astype(np.float32), arrays['device_context_y'])
                np.testing.assert_array_equal(estimator.classes_, self.state['device_context']['classes'])
            # 单行服务会多次查询同一组设备训练样本。沿用主教师的 CPU 后端，
            # 只复用这些固定样本的注意力投影，每条新读数仍执行完整预测。
            self._device_teacher = _prediction_teacher(estimator)
        if len(matched) <= 32:
            return self._device_teacher.predict_proba(matched.astype(np.float32))
        return np.concatenate([self._device_teacher.predict_proba(matched[start:start + 32].astype(np.float32))
                               for start in range(0, len(matched), 32)], axis=0)

    def _predict_device(self, raw, probabilities, *, device_cache=None):
        module = self._device_module()
        arrays = self.auxiliary_arrays
        if self._device_backgrounds is None:
            self._device_backgrounds = (module.background(arrays['device_context_x']),
                                        module.background(arrays['classifier_source_x']))
        selected = module.supported(module.background(raw), *self._device_backgrounds)
        if not selected.any():
            if len(raw) > 1:
                self._device_batch_outputs.clear()
            return probabilities
        labels = np.asarray(self.state['device_context']['classes'])
        matched = raw[selected]
        if device_cache is not None:
            # 缓存只含冻结设备教师的条件概率。标签顺序、训练张量摘要、资产、执行模式和有序查询
            # 共同确定其身份；候选自身的父网概率及混合权重每次重新计算。
            namespace = json.dumps(['gas-device-evaluation-v1', self.teacher_state, self.state['device_context'],
                                    os.environ.get('ZHIXUN_SDK_PREDICT_THREADS')], sort_keys=True)
            local, hit = frozen_batch_values(device_cache, namespace, matched, self._device_probabilities, width=len(labels))
            self.round_cache_hits['device_evaluation'] = self.round_cache_hits.get('device_evaluation', 0) + int(hit)
        elif len(raw) == 1:
            saved = self._device_batch_outputs.get(matched[0].tobytes())
            local = saved[None, :] if saved is not None else self._device_probabilities(matched)
        else:
            local = self._device_probabilities(matched)
        if len(raw) > 1:
            self._device_batch_outputs = {row.tobytes(): probability for row, probability in zip(matched, local, strict=True)}
        positions = np.ix_(selected, labels)
        result = probabilities.copy()
        result[positions] = local * probabilities[positions].sum(axis=1,keepdims=True)
        return result

    def _check_context(self):
        spec=self.state.get('baseline_context')
        fields={'baseline_context_x','baseline_context_y'} & self.auxiliary_arrays.keys()
        if spec is None and not fields:
            return
        if (not isinstance(spec,dict) or spec.get('format') not in BASELINE_ENSEMBLES
                or fields!={'baseline_context_x','baseline_context_y'}
                or sum(BASELINE_ENSEMBLES[spec['format']]) != self.teacher_state['n_estimators']):
            raise ValueError('气体本底上下文描述与产物不符')
        x,y=self.auxiliary_arrays['baseline_context_x'],self.auxiliary_arrays['baseline_context_y']
        if (x.dtype!=np.float32 or y.dtype!=np.int64 or not baseline_context_available(x,y) or not np.isfinite(x).all()
                or len(x)!=spec['rows'] or sha256(x.tobytes()).hexdigest()!=spec['features_sha256']
                or sha256(y.tobytes()).hexdigest()!=spec['labels_sha256']):
            raise ValueError('气体本底上下文的来源或数据无效')

    def add_baseline_context(self,source,labels):
        if self._teacher is not None or self._prediction_cache or self.state.get('baseline_context'):
            raise ValueError('只能在尚未拟合的新候选上首次接入本底上下文')
        alternate=baseline_context(source,labels)
        y=np.asarray(labels,dtype=np.int64).copy()
        self.auxiliary_arrays.update(baseline_context_x=alternate,baseline_context_y=y)
        # 将组合规模写入模型，旧产物重载后仍按原先的六加二配置预测。
        format_name = next(name for name, counts in BASELINE_ENSEMBLES.items()
                           if sum(counts) == self.teacher_state['n_estimators'])
        self.state['baseline_context']={'format':format_name,'rows':len(y),
              'features_sha256':sha256(alternate.tobytes()).hexdigest(),'labels_sha256':sha256(y.tobytes()).hexdigest(),
              'source_features_sha256':sha256(np.asarray(source,dtype=np.float64).tobytes()).hexdigest()}
        self.auxiliary_arrays['complete_state']=np.asarray(json.dumps(self.state,sort_keys=True,separators=(',',':'),allow_nan=False))
        self._check_context()

    def _check_drift_readout(self):
        spec=self.state.get('drift_readout')
        coeff=self.auxiliary_arrays.get('drift_coefficients')
        if spec is None and coeff is None:
            return
        classes=self.base.meta['classes']
        if (not isinstance(spec,dict) or spec.get('format')!='gas-drift-pair-v1'
                or 'Ethanol' not in classes or 'Acetaldehyde' not in classes
                or spec.get('reported')!=classes.index('Ethanol') or spec.get('target')!=classes.index('Acetaldehyde')
                or coeff is None or coeff.dtype!=np.float64 or coeff.shape!=(97,) or not np.isfinite(coeff).all()
                or sha256(coeff.tobytes()).hexdigest()!=spec.get('coefficients_sha256')):
            raise ValueError('气体漂移读出描述与实际权重不符')

    def predict_batch(self, raw):
        return self.predict(self._prepare_evaluation_batch(raw), device_cache=self._round_cache)

    def predict(self,raw, *, device_cache=None):
        probabilities=super().predict(raw)
        spec=self.state.get('drift_readout')
        if spec is not None:
            if __package__:
                from .gas_drift_readout import apply_readout
            else:
                from gas_drift_readout import apply_readout
            probabilities=apply_readout(raw,probabilities,spec,self.auxiliary_arrays['drift_coefficients'])
        if 'toluene_readout' in self.state:
            if __package__:
                from .gas_toluene_readout import apply_readout as apply_toluene
            else:
                from gas_toluene_readout import apply_readout as apply_toluene
            probabilities=apply_toluene(raw,probabilities,self.state['toluene_readout'],self.auxiliary_arrays)
        if 'kinetic_readout' in self.state:
            if __package__:
                from .gas_kinetic_support import apply_readout as apply_kinetics
            else:
                from gas_kinetic_support import apply_readout as apply_kinetics
            probabilities=apply_kinetics(raw,probabilities,self.state['kinetic_readout'],self.auxiliary_arrays)
        if 'device_context' in self.state:
            probabilities = self._predict_device(np.asarray(raw), probabilities, device_cache=device_cache)
        return probabilities

    def bind_round_cache(self, folder):
        """仅训练宿主可绑定轮内临时目录；逐行交付入口仍从实物模型预测。"""
        self._round_cache = folder

    def _teacher_probabilities(self, rows):
        # 保持训练用的教师目标不变。评价和单行服务对同一类成品采用相同的提前作答规则。
        if ((self.state.get('baseline_context') or {}).get('format') != 'gas-baseline-3plus1-v1'
                or 'device_context' in self.state):
            return super()._teacher_probabilities(rows)
        self._load_teacher()
        alternate = self._teacher.alternate.predict_proba(rows)
        uncertain = alternate.max(axis=1) < 0.85
        if uncertain.any():
            alternate = alternate.copy()
            alternate[uncertain] = (.75 * self._teacher.raw.predict_proba(rows[uncertain])
                                    + .25 * alternate[uncertain])
        return alternate

    def _prefetch_evaluation(self, raw):
        # 部分行已预取时仍走原分批方式，防止按不同分组算出的浮点数混入整批复用文件。
        if self._round_cache is None or not len(raw) or any(row.tobytes() in self._prediction_cache for row in raw):
            return super()._prefetch_evaluation(raw)
        # 直接绑定实际源张量，矩阵或源标签被替换时不能沿用仅同名的教师配置。
        source_arrays = {key:(list(self.auxiliary_arrays[key].shape),
                              self.auxiliary_arrays[key].dtype.str,
                              sha256(self.auxiliary_arrays[key].tobytes()).hexdigest())
                         for key in ('classifier_source_x','classifier_source_y','baseline_context_x','baseline_context_y')
                         if key in self.auxiliary_arrays}
        namespace = json.dumps(['classifier-evaluation-confident-member-v2','device_context' in self.state,
                                os.environ.get('ZHIXUN_SDK_PREDICT_THREADS'),
                                self.meta['classes'],self.teacher_state,self.state.get('baseline_context'),source_arrays],sort_keys=True)
        hit = frozen_teacher_values(self._round_cache, namespace, raw, self._prediction_cache,
                                    self._teacher_probabilities, width=len(self.meta['classes']))
        self.round_cache_hits['evaluation'] += int(hit)

    def _load_teacher(self):
        if self._teacher is not None:return
        import torch
        torch.set_num_threads(_teacher_threads())
        torch.manual_seed(42)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_NO_BROWSER']='1'
        # skrub默认缓存指向用户目录；固定到本次临时工作目录，退出后不留副本。
        with TemporaryDirectory(prefix='harness-tabpfn-classifier-') as temporary:
            os.environ['SKB_DATA_DIRECTORY']=temporary
            package=author_package()
            path=Path(temporary)/'tabpfn-v3.5-fast-20260909.safetensors'
            restore_teacher(self.auxiliary_arrays['teacher_asset'],path)
            def fitted(prefix,count,seed):
                torch.manual_seed(seed)
                predictor=package.TabPFNClassifier(model_path=path,n_estimators=count,device='cpu',random_state=seed,
                    fit_mode='fit_with_cache',n_preprocessing_jobs=1,inference_precision=torch.float32)
                predictor.fit(self.auxiliary_arrays[prefix+'_x'].astype(np.float32),self.auxiliary_arrays[prefix+'_y'])
                if not np.array_equal(predictor.classes_,np.arange(len(self.meta['classes']))):
                    raise ValueError('教师类别顺序不匹配')
                return _prediction_teacher(predictor)
            if self.state.get('baseline_context'):
                raw_count, alternate_count = BASELINE_ENSEMBLES[self.state['baseline_context']['format']]
                model=_BaselineEnsemble(fitted('classifier_source',raw_count,42),fitted('baseline_context',alternate_count,43))
            else:
                model=fitted('classifier_source',self.teacher_state['n_estimators'],42)
        self._teacher=model

    def _combine(self,logits,teacher):
        return arithmetic_output(logits,teacher,self.state['parent_weight'])


def from_classifier_asset(model,asset_path,*,parent_weight=.25):
    base=classification_base(model)
    with np.load(asset_path,allow_pickle=False) as saved:
        fields=['teacher_asset','teacher_license','classifier_source_x','classifier_source_y','tabpfn_classifier_state']
        arrays={field:saved[field].copy() for field in fields}
    teacher=json.loads(str(arrays['tabpfn_classifier_state']))
    state={'format':FORMAT,'kind':KIND,'parent_weight':float(parent_weight),
        'training_objective':'arithmetic_prediction_before_frozen_source_repair',
        'teacher_parameters':teacher['teacher_parameters'],'teacher_sha256':teacher['asset_sha256'],
        'source_context_rows':len(arrays['classifier_source_y']),'feature_view':'raw',
        'context_updates':'frozen_source_context','source_repair':'original_parent_metadata'}
    arrays['complete_state']=np.asarray(json.dumps(state,sort_keys=True,separators=(',',':'),allow_nan=False))
    return TabPFNClassifierModel(base,arrays)
