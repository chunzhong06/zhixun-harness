"""从正式运行账本生成按任务查阅的索引；不训练、不补传、不改原记录。"""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/"results/experiments"
GROUPS=("官方气体与反馈","轴承寿命","发动机与飞行","环境传感器与现场驱动","风机功率","跨任务与通用机制")


def classify(protocol,title):
    value=(Path(protocol).stem+" "+title).lower()
    def has(words):
        return any((word in value if not word.isascii() else
                    re.search(r"(?<![a-z])"+re.escape(word)+r"(?![a-z])",value)) for word in words)
    found=[]
    if has(("gas-turbine","燃机","engine","flight","cmapss","ncmapss","发动机","航班")):found.append(GROUPS[2])
    if has(("bearing","pronostia","轴承","kaist","ims")):found.append(GROUPS[1])
    if has(("wind","kelmarsh","风机","叶轮")):found.append(GROUPS[4])
    if has(("london","station","beijing","air","aircubes","plume","vapor","chip","sensor","ethylene","sgp40","ds-ps","smell","探头","芯片","标气","空气仪","温控")):found.append(GROUPS[3])
    if has(("gas","气体")) and GROUPS[3] not in found and "gas-turbine" not in value:found.append(GROUPS[0])
    return found[0] if len(found)==1 else GROUPS[5]


def load(path):
    if not path.is_file():return {}
    if path.suffix.lower()!=".json":
        # 早期冻结协议采用Markdown，标题可导航，不能当成JSON或缺失记录。
        body=path.read_text(encoding="utf-8-sig")
        heading=re.search(r"(?m)^#+\s+(.+)$",body)
        return {"name":heading[1] if heading else path.stem}
    return json.loads(path.read_bytes())


def link(path,label):
    p=ROOT/path
    if not p.exists():return label+"（原路径已不可用）"
    return f"[{label}]({os.path.relpath(p,DEST).replace(chr(92),'/')})"


def build():
    ledger=load(ROOT/"research-state.json");rows=[]
    for run in ledger["runs"]:
        protocol=run.get("protocol","");payload=load(ROOT/protocol)
        title=payload.get("中文名称") or payload.get("name") or Path(protocol).stem
        title=str(title).replace("|","／").replace("\n"," ")
        folder=run.get("folder","")
        cloud=load(ROOT/"results/engineering/research-workflow/swanlab"/run["id"]/"cloud.json")
        rows.append({"id":run["id"],"topic":classify(protocol,title),"name":title,"kind":run["kind"],
                     "status":run["status"],"seconds":run["elapsed_seconds"],"protocol":protocol,
                     "receipt":folder+"/run.json","protocol_available":(ROOT/protocol).is_file(),
                     "receipt_available":(ROOT/folder/"run.json").is_file(),
                     "cloud_verified":cloud.get("cloud_verified",False),"cloud_url":cloud.get("url")})
    fingerprint=hashlib.sha256(json.dumps(ledger["runs"],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    data={"source":"research-state.json","runs_fingerprint":fingerprint,"run_count":len(rows),"runs":rows}
    DEST.mkdir(exist_ok=True)
    (DEST/"index.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    text=["# 实验运行索引\n\n从[研究账本](../../research-state.json)生成，按研究对象组织全部已登记作业，失败和核验作业一并保留。运行编号只用于追溯；一项作业可以含多个模型与对照，不等于一轮比赛优化。实际结论从[实验结果总览](local-defaults/实验结果.md)进入。\n\n",
          f"共 {len(rows)} 项登记；刷新命令：`python -B engineering/cli/workspace/build_research_index.py`。[结构化索引](index.json)\n\n"]
    labels={"succeeded":"完成","failed":"失败","failed_before_start":"启动前失败","timed_out":"超时","aborted":"中止","running":"运行中"}
    for topic in GROUPS:
        members=[r for r in rows if r['topic']==topic]
        text.append(f"## {topic}\n\n| 运行 | 原作业名称 | 状态 | 秒 | 协议与原回执 | 云端核验 |\n|---|---|---|---:|---|---|\n")
        for row in members:
            cloud=f"[已核验]({row['cloud_url']})" if row['cloud_verified'] and row['cloud_url'] else "未记为已核验"
            refs=link(row['protocol'],'协议')+" · "+link(row['receipt'],'回执')
            text.append(f"| {row['id']} | {row['name']} | {labels.get(row['status'],row['status'])} | {row['seconds']:.2f} | {refs} | {cloud} |\n")
        text.append("\n")
    (DEST/"README.md").write_text(''.join(text),encoding='utf-8')
    print(json.dumps({"runs":len(rows),"topics":{g:sum(r['topic']==g for r in rows) for g in GROUPS},
                      "missing_receipts":sum(not r['receipt_available'] for r in rows)},ensure_ascii=False))


if __name__=="__main__":build()
