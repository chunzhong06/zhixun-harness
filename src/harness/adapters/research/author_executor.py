"""作者源码的本地执行边界：原始失败保留，探索评分由外层真实预测重算。"""
from hashlib import sha256
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from harness.adapters.providers.research_broker import ACTOR
from harness.learning.models.sparse_reference import correct,setting_of
from harness.adapters.storage.local import atomic_json
from harness.adapters.research.runtime_resources import BudgetExhausted, load_runtime_assets

ROOT = __import__("harness.paths", fromlist=["workspace_root"]).workspace_root()


class AuthorExecutor:
    def __init__(self,folder,workspace,oracle,*,timeout=90,scorer=None,runtime_assets=None,execution_budget=None,
                 candidate_feedback=None):
        self.folder=Path(folder).resolve();self.workspace=Path(workspace).resolve()
        self.oracle=oracle;self.timeout=timeout;self.records=[];self.phase='exploration'
        self.scorer=scorer;self.diagnostic=False;self.pending_replay=[]
        # 宿主先核验；子进程仍会在安装审计前重新核验，不给生成程序修改清单的接口。
        load_runtime_assets(runtime_assets)
        self.runtime_assets=Path(runtime_assets).resolve() if runtime_assets is not None else None
        self.execution_budget=execution_budget
        self.candidate_feedback=candidate_feedback

    def add_candidate_feedback(self, path, record):
        """硬预测评分及身份核验后调用宿主钩子；额外反馈不覆盖主分数或身份。

        钩子只接收已保存候选 JSON 路径。宿主负责限定可返回的信息，尤其不能
        将确认标签或确认分数回传作者；普通程序和 CoMind 单元格共用此入口。
        """
        if self.candidate_feedback is None:
            return ''
        feedback = self.candidate_feedback(Path(path))
        if not isinstance(feedback, dict):
            raise TypeError('candidate_feedback 必须返回字典')
        encoded = json.dumps(feedback, ensure_ascii=False, allow_nan=False)
        record['additional_feedback'] = json.loads(encoded)
        return '\nAdditional candidate feedback: ' + encoded + '\n'

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
        script.write_text(code_text,encoding='utf-8')
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env['TMP']=env['TEMP']=str(dest/'tmp');(dest/'tmp').mkdir()
        command=[sys.executable,'-B','-m','harness.cli.execute_author_program','--script',str(script)]
        if self.runtime_assets is not None:command+=['--runtime-assets',str(self.runtime_assets)]
        limit=min(self.timeout,exec_timeout)
        slot=self.execution_budget.execution(limit) if self.execution_budget is not None else nullcontext(SimpleNamespace(timeout_seconds=limit))
        elapsed=0.
        try:
            with slot as lease:
                start=time.monotonic()
                try:
                    process=subprocess.run(command,cwd=dest,env=env,capture_output=True,text=True,encoding='utf-8',timeout=lease.timeout_seconds)
                    result={'returncode':process.returncode,'stdout':process.stdout,'stderr':process.stderr}
                except subprocess.TimeoutExpired as exc:
                    result={'returncode':1,'stdout':exc.stdout.decode('utf-8',errors='replace') if isinstance(exc.stdout,bytes) else (exc.stdout or ''),
                            'stderr':'生成程序超过限定执行时间'}
                finally:
                    elapsed=time.monotonic()-start
        except BudgetExhausted as exc:
            result={'returncode':1,'stdout':'','stderr':str(exc)}
        result['execution_time']=elapsed
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
                    additional = self.add_candidate_feedback(path, record)
                    candidates.append(record)
                    result['stdout']=result['stdout'].replace('HARNESS_EVALUATION:'+path.stem,record['label']+' | 合法探索损失: '+str(score))
                    result['stdout']+=additional
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
