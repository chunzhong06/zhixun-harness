"""用同一份首轮原始回复隔离 row_filter 包装歧义，后九轮按新父版本重新提案。"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from engineering.cli.study import compare_luna_gas as gas


def watch():
    original = gas.DEFAULT
    branch = original.with_name("gas-format")
    branch.mkdir(parents=True, exist_ok=True)
    with gas.LocalStore(branch).exclusive():
        env = gas.prepare(branch)
        # 通用语义转换：只解开单键动作包装；内部字段仍经原实验同一套验证。
        strict = gas.read_response
        def read_response(path, req):
            actions, reason = strict(path, req)
            return [a["row_filter"] if isinstance(a, dict) and set(a) == {"row_filter"}
                    and isinstance(a["row_filter"], dict) else a for a in actions], reason
        gas.read_response = read_response
        first = gas.exchange(original, "responses", "mature", 1)
        duplicate = gas.exchange(branch, "responses", "mature", 1)
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        if not duplicate.exists():
            duplicate.write_bytes(first.read_bytes())
        if gas.file_sha(first) != gas.file_sha(duplicate):
            raise ValueError("首轮须重放原始回复，不更换建议")
        records = []
        for n in range(1, 11):
            while (next_record := gas.run_round(branch, "mature", n, env, records)) is None:
                time.sleep(1)
            records.append(next_record)
        # 其他三组已执行一次，按内容摘要引用同一份预测和末版；不再驱动或训练。
        other = ("original", "fixed", "project")
        while not all((original/"rounds"/arm/"10.json").exists() for arm in other):
            time.sleep(1)
        histories = {arm:[gas.read(original/"rounds"/arm/f"{n:02d}.json") for n in range(1,11)] for arm in other}
        for rs in histories.values():
            key = rs[-1]["served"]
            source, dest = (folder/"models"/(key+".npz") for folder in (original, branch))
            if not dest.exists():
                os.link(source, dest)
            if gas.file_sha(dest) != key:
                raise ValueError("借用的末版摘要不符")
        gas.final_score(branch, env, {**histories, "mature":records})


if __name__ == "__main__":
    watch()
