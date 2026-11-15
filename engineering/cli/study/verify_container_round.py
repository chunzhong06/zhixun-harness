"""真实Linux容器两轮权重交接和独立预测核验；不调用模型服务。"""
from pathlib import Path
from hashlib import sha256
import json
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]

def main():
    task=ROOT/'scratch/contest-adaptation-20261115/container-task'
    out=ROOT/'results/engineering/contest-continuation/container-verification'
    out.mkdir(parents=True,exist_ok=False)
    def linux(path):return '/mnt/'+str(path)[0].lower()+str(path)[2:].replace('\\','/')
    def run(argv):
        r=subprocess.run(['wsl','-d','Ubuntu','-u','root','--','docker',*argv],capture_output=True,timeout=300)
        if r.returncode:raise RuntimeError(r.stderr.decode('utf8','replace')[-3000:])
        return r.stdout
    common=['run','--rm','--network','none','--cpus','4','--memory','8g','-v',linux(task)+':/input:ro',
       '-v',linux(ROOT/'data/official')+':/data:ro','-v',linux(out)+':/output','--entrypoint','python','zhixun-harness:trial-20261115']
    parent=task/'initial.npz'
    r=run(common+['-B','engineering/cli/workflow/native_continuation.py','--task-dir','/input','--parent','/input/initial.npz',
       '--expected-parent-sha256',sha256(parent.read_bytes()).hexdigest(),'--program','/input/program.py','--output','/output/round-1','--max-epochs','60'])
    (out/'round-1.log').write_bytes(r)
    (task/'next.py').write_text('def train(session):\n    session.train_epoch(learning_rate=.0001,trainable=["w2","b2"])\n',encoding='utf8')
    checkpoint=out/'round-1/checkpoint.npz'
    r=run(common+['-B','engineering/cli/workflow/native_continuation.py','--task-dir','/input','--parent','/output/round-1/checkpoint.npz',
       '--expected-parent-sha256',sha256(checkpoint.read_bytes()).hexdigest(),'--program','/input/next.py','--output','/output/round-2','--max-epochs','1'])
    (out/'round-2.log').write_bytes(r)
    a=json.loads((out/'round-1/training.json').read_bytes());b=json.loads((out/'round-2/training.json').read_bytes())
    assert b['loaded_tensors']==a['final_tensors']
    assert b['parent_checkpoint_sha256']==a['checkpoint_sha256']
    assert b['final_steps']>b['initial_steps']
    host=ROOT/'results/experiments/contest-continuation/official-complete-bootstrap-v2/candidate/predictions.npz'
    with np.load(host) as h,np.load(out/'round-1/predictions.npz') as l:
        differences={k:float(np.max(np.abs(h[k]-l[k]))) for k in h.files}
    image=json.loads(run(['image','inspect','zhixun-harness:trial-20261115']))[0]
    result={'real_linux_container':True,'platform':image['Os']+'/'+image['Architecture'],'image_id':image['Id'],
            'image_bytes':image['Size'],'two_round_parent_inheritance':True,'first_round_steps':a['final_steps']-a['initial_steps'],
            'second_round_steps':b['final_steps']-b['initial_steps'],'host_vs_linux_max_prediction_difference':differences,
            'round1_artifact_bytes':a['artifact_bytes'],'round2_artifact_bytes':b['artifact_bytes']}
    (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
