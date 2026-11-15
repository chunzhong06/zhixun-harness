"""从官方原包重建完整发动机，与旧强父版在同一官方及迁移输入上比较。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from hashlib import sha256
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
import numpy as np
from engineering.cli.workflow.native_continuation import execute
from engineering.cli.study.evaluate_native_transfer import dataset
from harness.integrations.native_research import prepare, export_release
from harness.models.complete_checkpoint import load_checkpoint
from harness.models.official_rul_cpu import _trainer
from harness.runtime.storage import atomic_json


def main():
    out = Path(sys.argv[1]).resolve()
    task, oracle = prepare('engine', out)
    program = out/'bootstrap.py'
    program.write_text('def train(session):\n    session.build_regression_components((21,22,23),(0,2,0),epochs=60,head_epochs=12)\n', encoding='utf8')
    parent = task/'initial.npz'
    report = execute(task, parent, program, out/'candidate', expected_parent_sha256=sha256(parent.read_bytes()).hexdigest(), max_epochs=60)
    specification = json.loads((task/'task.json').read_bytes())
    current = json.loads((ROOT/'engineering/configs/tasks/current-models.json').read_bytes())['tasks']['engine']
    oldpath = ROOT/current['checkpoint']
    assert sha256(oldpath.read_bytes()).hexdigest() == current['sha256']
    with ZipFile(specification['archive']) as z, TemporaryDirectory(dir=out) as temp:
        with _trainer(z,Path(temp),'C') as trainer:
            models = {'official': load_checkpoint(parent,trainer.Model), 'old_complete':load_checkpoint(oldpath,trainer.Model),
                      'rebuilt':load_checkpoint(out/'candidate/checkpoint.npz',trainer.Model)}
            raw = [json.loads(row) for row in z.read('public/validation.jsonl').splitlines()]
            from importlib import import_module
            data = import_module(trainer.__package__+'.data')
            x = np.asarray([data.flatten(row['inputs'])[1] for row in raw])
            # 官方目标从既有读取器获取，避免猜测JSON标签字段。
            from engineering.cli.study.evaluate_rul_baseline import PACKAGES
            package = PACKAGES['engine'][2](Path(specification['archive']))
            y = np.asarray([r.label for r in package.validation])
            tx, ids, groups, truth = dataset('engine')
            outputs = {name: np.asarray([m.predict(row[None,:])[0] for row in x]) for name,m in models.items()}
            transfer = {name: np.asarray([m.predict(row[None,:])[0] for row in tx]) for name,m in models.items()}
    np.savez_compressed(out/'predictions.npz',y=y,**outputs)
    np.savez_compressed(out/'transfer.npz',ids=np.asarray(ids),y=truth(),**transfer)
    rmse=lambda a,b:float(np.sqrt(np.mean((np.asarray(a)-b)**2)))
    result={'official':{k:rmse(y,v) for k,v in outputs.items()},'transfer':{k:rmse(truth(),v) for k,v in transfer.items()},
        'actual_parent_inherited':report['loaded_tensors']==report['migration']['retained_parent_tensors'],
        'base_gradient_steps':report['final_steps']-report['initial_steps'],'component_history':report['component_history'],
        'training_seconds':report['training_seconds'],'checkpoint_bytes':report['checkpoint_bytes'],
        'official_rows':len(y),'transfer_rows':len(tx)}
    export_release(task,out/'candidate/checkpoint.npz',out/'release')
    atomic_json(out/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
