"""只在已开分的2024资料上检查强起点、拒绝换版和反馈身份的完整一致性。"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"engineering/src")]
from engineering.cli.workflow import london_field_driver as scenario
from engineering.cli.study.verify_london_feedback import published_trace,score_with_gate,first_packet
from harness.models.sparse_reference import setting_of
from harness.models.station_residual import Rule


def check_parameter_aliases():
    """有输入和常量两条执行路径均须接受显式等价默认值。"""
    x=np.tile(np.arange(9,dtype=float),(24,1));y=np.arange(24,dtype=float)
    fit=np.ones(24,dtype=bool)
    for flexible in (False,True):
        tools=scenario.StationTools(x,scenario.DONORS,scenario.SEASON,flexible=flexible)
        for kind in ("bias","network"):
            proposal={"hypothesis":"核对等价默认值","recipe":{"background":"BL0","correction":kind,
                      "days":21,"strength":.5,"parameters":{"alpha":8.}}}
            explicit=tools.parse(proposal);implicit=Rule("BL0",kind,21,.5)
            actual=tools.issue(tools.train(explicit,fit,y),fit)
            expected=tools.issue(tools.train(implicit,fit,y),fit)
            assert np.array_equal(actual,expected)


def main():
    check_parameter_aliases()
    scenario.configure("2024-07-01")
    previous=ROOT/"results/experiments/local-defaults/whole-maintenance/london-learner-equivalence"
    scores=json.loads((previous/"scores.json").read_bytes())
    spec=json.loads((ROOT/"engineering/configs/experiments/london-strong-initial.json").read_bytes())
    setting=setting_of(spec["feedback"])
    tools=scenario.prepare(ROOT/"data/public/london-air-2024/stations.zip",True,True)
    tools.static=Rule(**spec["recipe"])
    checked=0
    for station in scores["counts"]:
        ref=scenario.readings(ROOT/"data/public/london-air-2024/stations.zip",station).to_numpy()
        y,fit,windows,service,returned=scenario.roles(ref)
        model=tools.train(tools.static,fit,y)
        actual=scenario.fixed_stream(tools.issue(model,np.arange(len(y)))[:,None],y,windows,service,returned,setting,True)[:,0]
        independent=published_trace(tools,model,setting,y,windows,service,returned,True)
        with np.load(previous/station/"fixed-winner.npz") as archive:expected=archive["prediction"]
        assert np.allclose(actual,expected,rtol=0,atol=1e-10)
        assert np.allclose(actual,independent,rtol=0,atol=1e-10)
        # 相同候选无法严格获胜，必须保持包含原反馈状态的父版。
        rejected=score_with_gate(tools,tools.static,setting,y,fit,windows,service,returned,warm=True,parent_feedback=setting)
        assert np.array_equal(rejected,independent)
        packet=first_packet(tools,model,model,setting,y,windows,setting)
        assert packet["initial"]==0 and packet["models"][0][1]==setting
        checked+=1
    print(json.dumps({"初始完整预测精确对应既有最强固定的站点":checked,"独立重放最大差":0.0},ensure_ascii=False))


if __name__=="__main__":main()
