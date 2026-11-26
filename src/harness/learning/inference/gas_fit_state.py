"""在单线程预测进程中，复用与已核验教师和源张量完全相同的原始集成拟合状态。"""
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
from shutil import copyfileobj
from zipfile import ZipFile

DIRECTORY = Path('/opt/zhixun/gas-fit-v1')

def load_raw(model, count, seed, package, weights):
    if os.environ.get('ZHIXUN_SDK_PREDICT_THREADS') != '1' or not (DIRECTORY/'manifest.json').is_file():
        return None
    info = json.loads((DIRECTORY/'manifest.json').read_bytes())
    if (count != info['count'] or seed != info['seed'] or model.teacher_state['asset_sha256'] != info['teacher']
            or model.meta['classes'] != info['classes'] or model.meta['paths'] != info['paths']):
        return None
    arrays = model.auxiliary_arrays
    for axis in ('x','y'):
        tensor = arrays['classifier_source_'+axis]
        if [tensor.dtype.str,list(tensor.shape),sha256(tensor.tobytes()).hexdigest()] != info[axis]:
            return None
    artifact = DIRECTORY/'raw.tabpfn_fit'
    with artifact.open('rb') as f:
        if file_digest(f,'sha256').hexdigest() != info['sha256']:
            return None
        f.seek(0)
        bound = weights.with_suffix('.tabpfn_fit')
        with ZipFile(f) as source, ZipFile(bound,'x') as out:
            params=json.loads(source.read('init_params.json'))
            params['model_path']=str(weights.resolve())
            out.writestr('init_params.json',json.dumps(params))
            for name in ('fitted_attrs.joblib','executor_state.joblib'):
                with source.open(name) as reader,out.open(name,'w') as writer:
                    copyfileobj(reader,writer,1024*1024)
    return package.TabPFNClassifier.load_from_fit_state(bound,device='cpu')
