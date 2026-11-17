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
    from .probability_mixture import arithmetic_output
    from .quantized_teacher import restore_teacher
    try:
        from .complete_checkpoint import FORMAT
    except ImportError:
        from .complete_model import FORMAT
else:
    from tabular_classifier import TabICLClassifierModel,classification_base
    from probability_mixture import arithmetic_output
    from quantized_teacher import restore_teacher
    from complete_model import FORMAT

KIND='tabpfn-classification-parent'


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
        self._teacher=None;self._training_teacher={};self._prediction_cache={};self._repair=None
        self.validate()

    def _load_teacher(self):
        if self._teacher is not None:return
        import torch
        torch.set_num_threads(1);torch.manual_seed(42)
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TABPFN_NO_BROWSER']='1'
        # skrub默认缓存指向用户目录；固定到本次临时工作目录，退出后不留副本。
        with TemporaryDirectory(prefix='harness-tabpfn-classifier-') as temporary:
            os.environ['SKB_DATA_DIRECTORY']=temporary
            package=author_package()
            path=Path(temporary)/'tabpfn-v3.5-fast-20260909.safetensors'
            restore_teacher(self.auxiliary_arrays['teacher_asset'],path)
            model=package.TabPFNClassifier(model_path=path,n_estimators=self.teacher_state['n_estimators'],
                device='cpu',random_state=42,fit_mode='fit_with_cache',n_preprocessing_jobs=1,
                inference_precision=torch.float32)
            model.fit(self.auxiliary_arrays['classifier_source_x'].astype(np.float32),
                      self.auxiliary_arrays['classifier_source_y'])
        if not np.array_equal(model.classes_,np.arange(len(self.meta['classes']))):
            raise ValueError('教师类别顺序不匹配')
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
