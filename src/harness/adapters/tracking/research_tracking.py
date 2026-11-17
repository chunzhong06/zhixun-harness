"""研发实验的显式云端记录；凭据由SDK用户目录提供，与赛题离线记录器分开。"""
import math
import json
from hashlib import sha256
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import urlopen

from harness.adapters.storage.local import atomic_json


class ResearchSwanLab:
    """只上传调用者提供的配置与指标；结束后从云端读回，保存实际同步状态。"""
    EVIDENCE_KEY = '执行过程/逐轮证据'
    MAX_EVIDENCE_BYTES = 16*1024**2

    def __init__(self,settings,folder:Path,*,name,config,kind="research",description=None,resume_id=None,
                 run_id=None,resume_mode=None):
        import swanlab
        if settings["mode"]!="online" or swanlab.__version__!=settings["sdk_version"]:
            raise ValueError("需要已核对版本的SwanLab在线记录")
        self.sdk=swanlab;self.settings=settings;self.folder=Path(folder);self.latest={};self.evidence=[]
        if resume_id is not None and (run_id is not None or resume_mode is not None):
            raise ValueError('恢复身份只能指定一种形式')
        requested_id=resume_id or run_id
        probe=swanlab.Settings.Probe(hardware=False,runtime=False,requirements=False,conda=False,
                                    git=False,swanlab=False,monitor=False)
        opts=swanlab.Settings(interactive=False,probe=probe,terminal=swanlab.Settings.Terminal(proxy_type="none"))
        self.run=swanlab.init(project=settings["project"],workspace=settings["workspace"],name=name,
            description=description,job_type=kind,mode="online",on_online_error="raise",log_dir=str(self.folder/"sdk"),
            config=config,settings=opts,id=requested_id,resume='must' if resume_id else resume_mode)
        self.url=self.run.url;self.id=self.run.id
        if requested_id is not None and self.id != requested_id:
            raise ValueError('记录恢复没有绑定原云端作业')
        self._save("running",False)

    def _save(self,status,verified,**extra):
        record={"workspace":self.settings["workspace"],"project":self.settings["project"],"id":self.id,
                "url":self.url,"status":status,"cloud_verified":verified,"latest":self.latest,**extra}
        atomic_json(self.folder/"cloud.json",record)
        return record

    def log(self,metrics,step):
        clean={k:float(v) for k,v in metrics.items() if isinstance(v,(int,float,bool)) and math.isfinite(v)}
        if clean:self.run.log(clean,step=step);self.latest.update(clean)

    def record_evidence(self, value, step=None, upload=True):
        """仅接收本作业逐轮原始过程；中断恢复重复打印的同一证据只保留一次。"""
        if not isinstance(value, dict) or not isinstance(value.get('trace_id'), str):
            raise ValueError('运行证据缺少稳定身份')
        for previous in self.evidence:
            if previous['trace_id'] == value['trace_id']:
                if previous != value:
                    raise ValueError('同一轮运行证据在恢复后改变')
                return False
        # JSON往返保证保留的是当时快照，不受调用者后续修改影响。
        self.evidence.append(json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)))
        atomic_json(self.folder/'execution-evidence.json', self.evidence)
        text=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)
        if upload:
            self.run.log({self.EVIDENCE_KEY:self.sdk.Text(text,caption=f"{value.get('任务','任务')} 第{value.get('轮次','?')}轮")},
                         step=step if step is not None else len(self.evidence))
        return True

    @classmethod
    def read_remote_evidence(cls, remote):
        """取得云端正文，供最终核验或只补传缺失轮次；返回值不携带签名URL。"""
        series=[key for key in remote.series(metric_type='MEDIA',search=cls.EVIDENCE_KEY) if key.key==cls.EVIDENCE_KEY]
        if not series:return []
        if len(series)!=1:raise ValueError('云端出现重复的过程证据指标')
        records=series[0].metric(all=True).get('metrics',[])
        items=[item for record in records for item in record.get('items',[])]
        def fetch(item):
            url=item.get('url','')
            if not url.startswith('https://'):
                raise ValueError('云端证据缺少有效下载地址')
            with urlopen(url,timeout=20) as response:
                raw=response.read(cls.MAX_EVIDENCE_BYTES+1)
            if len(raw)>cls.MAX_EVIDENCE_BYTES:
                raise ValueError('单轮云端记录超出核验大小')
            return json.loads(raw)
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(fetch,items))

    def _verify_evidence(self, remote):
        """逐字段比对已下载正文；文件名或本地上传回执不代替核验。"""
        actual=self.read_remote_evidence(remote)
        if len(actual)!=len(self.evidence):return False
        expected={entry['trace_id']:entry for entry in self.evidence}
        if len({entry.get('trace_id') for entry in actual})!=len(expected):
            return False
        return all(expected.get(entry.get('trace_id'))==entry for entry in actual)

    def finish(self,success=True):
        """SDK完成与云端可读分开记录；核对本次标量和逐轮证据，不补传旧实验。"""
        evidence_sha = None
        if self.evidence:
            raw=json.dumps(self.evidence,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()
            evidence_sha=sha256(raw).hexdigest()
            # 完整证据使用逐轮Text文件，配置只放摘要，避免云端profile请求大小限制。
            self.run.config.update({'执行证据位置':self.EVIDENCE_KEY,'执行证据条数':len(self.evidence),'执行证据SHA256':evidence_sha})
        self.run.finish(state="success" if success else "crashed",async_log_timeout=30)
        path=f"{self.settings['workspace']}/{self.settings['project']}/{self.id}"
        missing=list(self.latest);remote_state="";error=None;evidence_verified=not self.evidence
        for attempt in range(5):
            try:
                remote=self.sdk.Api().run(path);remote_state=remote.state
                values=remote.summary(keys=list(self.latest))
                missing=[k for k,v in self.latest.items() if k not in values or
                    not isinstance(values[k].get("value"),(int,float)) or
                    not math.isclose(values[k]["value"],v,rel_tol=1e-8,abs_tol=1e-10)]
                if self.evidence:
                    config=remote.profile.get('config',{})
                    evidence_verified=(config.get('执行证据SHA256',{}).get('value')==evidence_sha and
                                       config.get('执行证据条数',{}).get('value')==len(self.evidence) and self._verify_evidence(remote))
                if not missing and evidence_verified and remote_state in ("FINISHED","CRASHED","ABORTED"):break
            except Exception as exc:error=type(exc).__name__
            if attempt<4:time.sleep(2)
        terminal=remote_state==("FINISHED" if success else "CRASHED")
        return self._save("finished" if success else "failed",not missing and terminal and evidence_verified,
                          remote_state=remote_state,missing_metrics=missing,verification_error=error,
                          evidence_count=len(self.evidence),evidence_sha256=evidence_sha,
                          evidence_cloud_verified=evidence_verified if self.evidence else None)
