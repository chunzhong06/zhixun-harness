"""作者源码的本地执行边界：原始失败保留，探索评分由外层真实预测重算。"""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from harness.integrations.research_broker import ACTOR
from harness.models.sparse_reference import correct,setting_of
from harness.runtime.storage import atomic_json

ROOT=Path(__file__).resolve().parents[4]


class AuthorExecutor:
    def __init__(self,folder,workspace,oracle,*,timeout=90,scorer=None):
        self.folder=Path(folder).resolve();self.workspace=Path(workspace).resolve()
        self.oracle=oracle;self.timeout=timeout;self.records=[];self.phase='exploration'
        self.scorer=scorer;self.diagnostic=False;self.pending_replay=[]

    def __call__(self,code_text,run_cwd,py_filepath,exec_timeout):
        """签名沿作者 code_util；训练不接触探索/确认标签，绝不接受模型自报分数。"""
        run_cwd=Path(run_cwd).resolve()
        if not run_cwd.is_relative_to(self.workspace) or Path(py_filepath).name!=py_filepath:
            raise ValueError('作者程序输出目录越界')
        source_path=run_cwd/py_filepath;source_path.write_text(code_text,encoding='utf-8')
        if self.pending_replay:
            previous=self.pending_replay.pop(0)
            if (sha256(code_text.encode()).hexdigest()!=previous['source_sha256'] or
                    previous['actor']!=ACTOR.get() or previous['diagnostic']!=self.diagnostic):
                raise ValueError('恢复程序、角色或动作类型与原实际执行不同')
            if previous['sequence']!=len(self.records)+1:raise ValueError('恢复改变执行顺序')
            self.records.append(previous);atomic_json(self.folder/'index.json',self.records)
            return {key:previous[key] for key in ('returncode','stdout','stderr','execution_time')}
        dest=self.folder/f'{len(self.records)+1:04d}';dest.mkdir(parents=True)
        shutil.copytree(run_cwd/'input',dest/'input');script=dest/py_filepath
        script.write_text(code_text,encoding='utf-8');start=time.monotonic()
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env['TMP']=env['TEMP']=str(dest/'tmp');(dest/'tmp').mkdir()
        try:
            process=subprocess.run([sys.executable,'-B',str(ROOT/'engineering/cli/study/execute_author_program.py'),'--script',str(script)],
                cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8',timeout=min(self.timeout,exec_timeout))
            result={'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
        except subprocess.TimeoutExpired as exc:
            result={'returncode':1,'stdout':exc.stdout.decode('utf-8',errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or ''),
                    'stderr':'生成程序超过限定执行时间'}
        result['execution_time']=time.monotonic()-start
        candidates=[]
        if result['returncode']==0:
            try:
                for path in sorted((dest/'artifacts').glob('candidate-*.json')):
                    record=json.loads(path.read_bytes());weights=path.with_suffix('.pkl')
                    if sha256(weights.read_bytes()).hexdigest()!=record['weights_sha256']:raise ValueError('模型权重身份变化')
                    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
                        raw=z['validation'];confirmation=z['confirmation']
                    if raw.shape!=self.oracle['validation_y'].shape or confirmation.shape!=self.oracle['confirmation_y'].shape:
                        raise ValueError('探索或确认预测行数不符')
                    if not np.isfinite(raw).all() or not np.isfinite(confirmation).all():raise ValueError('存在非有限预测')
                    if self.scorer:
                        measured=self.scorer(raw);score=measured['loss'];window_errors=[score]
                        record['task_metrics']=measured
                    else:
                        setting=setting_of(record['feedback'])
                        issued=correct(raw[:,None],self.oracle['validation_y'],self.oracle['validation_hours'],setting)[:,0]
                        window_errors=[]
                        for mask in self.oracle['exploration_masks']:
                            valid=mask&np.isfinite(self.oracle['validation_y'])
                            if not valid.any():raise ValueError('该作者任务缺少事前要求的探索参考')
                            window_errors.append(float(np.sqrt(np.mean((issued[valid]-self.oracle['validation_y'][valid])**2))))
                        score=float(np.sqrt(np.mean(np.square(window_errors))))
                    record.update(path=str(path),source_sha256=sha256(code_text.encode()).hexdigest(),exploration_rmse=window_errors,score=score)
                    candidates.append(record)
                    result['stdout']=result['stdout'].replace('HARNESS_EVALUATION:'+path.stem,record['label']+' | 合法探索损失: '+str(score))
                if not candidates and not self.diagnostic:raise ValueError('未通过harness_task.evaluate_model或submit_model交付实际模型')
                if candidates and self.diagnostic:raise ValueError('诊断动作不能偷偷提交新候选；需要选择实验动作')
                if ACTOR.get().startswith('submission'):
                    delivered=pd.read_csv(dest/'final/submission.csv')
                    if list(delivered.columns)!=['prediction'] or not np.allclose(delivered['prediction'].to_numpy(),confirmation,rtol=0,atol=1e-8):
                        raise ValueError('最终CSV与最后保存模型的确认预测不同')
                if candidates:result['stdout']+='\nFinal Validation Performance: '+str(candidates[-1]['score'])+'\n'
            except (ValueError,KeyError,OSError,TypeError) as exc:
                result['returncode']=1;result['stderr']+='\n外层交付核对：'+str(exc)
        record={'sequence':len(self.records)+1,'actor':ACTOR.get(),'phase':self.phase,'diagnostic':self.diagnostic,'directory':str(dest),
                'source':str(script),'source_sha256':sha256(code_text.encode()).hexdigest(),
                'input_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in (dest/'input').iterdir() if p.is_file()},
                'candidates':candidates,**result}
        atomic_json(dest/'execution.json',record);self.records.append(record)
        atomic_json(self.folder/'index.json',self.records)
        return result
