"""从第四次气体开发对照绘制论文图。只读既有预测、权重和公开训练资料。"""
import argparse
from io import BytesIO
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "engineering/src"))
from harness.data.official_gas import LABELS, read_official_gas
from harness.models.gas_views import predict_view, represent

RUN = ROOT / "results/experiments/official-gas/pilot-4"
PACKAGE = ROOT / "data/official/gas/cpu-a-guide-20261110-v1.zip"
TEAL = "#176C69"
ETHANOL = "#6576A0"
OTHER = "#DFE2E5"
RUST = "#B8694D"
INK = "#25363A"
GRAY = "#59696F"
HAIR = "#D7DFDE"


def read(path):
    return json.loads(path.read_bytes())


def setup_typeface(font):
    if font:
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        names = {f.name for f in font_manager.fontManager.ttflist}
        family = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC",
                                          "Source Han Sans SC", "PingFang SC") if name in names), None)
        if family is None:
            raise ValueError("请用 --font 指定可嵌入的中文字体")
    plt.rcParams.update({"font.family": family, "font.size": 8, "pdf.fonttype": 42,
                         "axes.unicode_minus": False, "axes.edgecolor": HAIR,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})


def example_evidence(folder):
    """按指定身份复读两例，核对展示分数的胜出类与正式保存的预测相同。"""
    package = read_official_gas(PACKAGE)
    held = read(folder / "protocol.json")["holdout"]
    original = read(folder.parent / "pilot-1/static/result.json")["holdout"]["predictions"]
    direct = read(folder / "pair-shape-knn/result.json")["holdout"]["predictions"]
    chosen = read(folder / "pair-shape-blend/result.json")["holdout"]["predictions"]
    by_id = {row.sample_id: row for row in package.validation}
    if set(held) != set(chosen) or set(original) != set(direct) or set(original) != set(chosen):
        raise ValueError("各方法的样本清单不相同")
    decided = read(folder / "comparison.json")["pair-shape-blend"]["decisions"]
    pair = decided[0]["pair"]
    source, target = pair["reported"], pair["target"]
    if (LABELS[source-1], LABELS[target-1]) != ("Ethanol", "Acetaldehyde"):
        raise ValueError("本图的类别标签不适用于该修正方向")
    labels = np.array([r.label - 1 for r in package.train])
    # 模型文件的标准化参数及两层权重来自随包检查点，与已采用版本的父模型一致。
    with np.load(BytesIO(package.checkpoint), allow_pickle=False) as m:
        mean, scale = m["mean"], m["scale"]
        bank = represent([r.values for r in package.train], "shape", mean, scale)
        examples = []
        for suffix in ("000658", "000007"):
            sid = f"gas_b02_{suffix}"
            row = by_id[sid]
            raw = np.array([row.values])
            x = represent(raw, "raw", mean, scale)
            logits = np.tanh(x @ m["w1"] + m["b1"]) @ m["w2"] + m["b2"]
            probs = np.exp(logits - logits.max())
            probs /= probs.sum(axis=1, keepdims=True)
            shape = represent(raw, "shape", mean, scale)
            # 复用真实读出，排序与相似距离的并列处理和运行工具保持一致。
            votes = predict_view({"method":"shape-knn", "train":bank, "labels":labels},
                                 raw, mean, scale, probs)
            blend = (probs + votes) / 2
            winner = int(blend.argmax()) + 1
            gated = target if original[sid] == source and winner == target else original[sid]
            if original[sid] != int(probs.argmax())+1 or gated != chosen[sid] or direct[sid] != target:
                raise ValueError(f"{sid} 的复读与原始结果不符")
            if chosen[sid] != row.label:
                raise ValueError("这两例应分别为正确修复及保住原正确预测")
            examples.append((sid, shape.reshape(16,8), np.vstack([probs, votes, blend])))
    return examples


def heading(fig, x, y, letter, words):
    fig.text(x, y, letter, size=10, weight="bold", color=TEAL, va="top")
    fig.text(x+.035, y, words, size=9.5, weight="bold", va="top")


def overview_figure(folder):
    """上图是公共运行路径，下图只读本次采用回执；技术箭头按实际消费者绘制。"""
    decisions = read(folder/"comparison.json")["pair-shape-blend"]
    adopted, = decisions["decisions"]
    split = read(folder/"protocol.json")["rounds"][adopted["round"]]
    result = read(folder/"pair-shape-blend/result.json")
    record = next(r for r in result["rounds"] if r["round"] == adopted["round"])
    fig = plt.figure(figsize=(7.95, 4.28))
    ax = fig.add_axes([.008,.018,.984,.964], xlim=(0,120), ylim=(0,85))
    ax.axis("off")
    def words(x,y,s,size=7.1,color=INK,align="center",bold=False):
        ax.text(x,y,s,ha=align,va="center",fontsize=size,color=color,
                weight="bold" if bold else "normal",zorder=7,linespacing=1.38)

    def box(x,y,w,h,s,*,face="white",edge=HAIR,bold=False,size=7.1):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.14,rounding_size=.63",
                                   lw=.65,edgecolor=edge,facecolor=face,zorder=4))
        words(x+w/2,y+h/2,s,size,bold=bold)

    def route(points,color=GRAY,dashed=False):
        xs,ys=zip(*points[:-1])
        if len(xs)>1:
            ax.plot(xs,ys,color=color,lw=.82,linestyle="--" if dashed else "-",zorder=2)
        a,b=points[-2:]
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=7,
                                    linewidth=.82,color=color,linestyle="--" if dashed else "-",
                                    shrinkA=0,shrinkB=1,zorder=3))

    words(1.8,81.4,"a",11,TEAL,"left",True)
    words(5.7,81.4,"工业模型的持续维护流程",11,INK,"left",True)
    # 资料用途独立。探索结果可调整下一次动作；确认资料只在提名冻结后打开。
    words(6.8,73.3,"在用模型",7.6,INK,bold=True)
    words(18.4,73.3,"探索样本",7.6,INK,bold=True)
    route([(6.8,71.3),(6.8,65.6),(11,65.6),(11,62.4)])
    route([(18.4,71.3),(18.4,65.6),(15,65.6),(15,62.4)])
    box(3.3,51.6,19.5,10.7,"核对预测与答案\n定位重复错误",face="#EEF2F7",edge="#AEBBCB",size=7.7)

    ax.add_patch(FancyBboxPatch((26,46.4),45.7,21.5,boxstyle="round,pad=.25,rounding_size=1.2",
                                facecolor="#EAF4F2",edgecolor="#A9C8C1",lw=.55,zorder=0))
    words(27.5,65.4,"本轮检查与试改",7.6,TEAL,"left",True)
    box(27.7,52.0,11.9,8.6,"维护计划",edge="#97B8B3",bold=True)
    box(42.3,52.0,11.4,8.6,"执行改法",edge="#97B8B3")
    box(56.4,52.0,13.1,8.6,"探索比较",edge="#97B8B3")
    route([(22.9,56.3),(27.6,56.3)])
    route([(39.7,56.3),(42.3,56.3)])
    route([(53.8,56.3),(56.4,56.3)])
    words(47.9,73.3,"合法训练资料",7.6,INK,bold=True)
    route([(47.9,71.3),(47.9,60.7)])
    words(62.9,73.3,"同批探索样本",7.6,INK,bold=True)
    route([(62.9,71.3),(62.9,60.7)])
    route([(62.9,51.9),(62.9,48.7),(33.7,48.7),(33.7,51.9)],TEAL)
    # 在线路中间压白局部底色，避免标签与返回线相叠。
    ax.text(47.9,48.7,"按探索结果调整下一步",ha="center",va="center",fontsize=6.2,color=TEAL,
            bbox={"facecolor":"#EAF4F2","edgecolor":"none","pad":1.3},zorder=8)
    words(75.3,60.1,"冻结提名",6.6,GRAY)
    route([(69.6,56.3),(80.1,56.3)],TEAL)
    words(89,73.3,"另置确认样本",7.6,INK,bold=True)
    route([(89,71.3),(89,62.4)])
    box(80.2,51.6,17.6,10.7,"原版与候选\n同样本确认",face="#F0EFF6",edge="#B9B2CD",size=7.5)
    route([(98,56.3),(101,56.3),(101,61.0),(103.9,61.0)],TEAL)
    route([(98,56.3),(101,56.3),(101,51.4),(103.9,51.4)])
    box(104,58.1,13.2,5.8,"改善：采用新版",face="#ECF5F2",edge="#A8C9BE",size=6.7)
    box(104,48.5,13.2,5.8,"否则：保留原版",size=6.7)
    route([(117.4,61),(118.6,61),(118.6,44.5),(114.5,44.5),(114.5,42.8)],TEAL)
    route([(110.6,48.3),(110.6,42.8)])
    box(101.9,35.2,15.8,7.5,"生效版本预测\n保存版本与输出",edge="#9FBAB8",bold=True,size=6.6)
    # 探索未提名新候选时直接保持当前版本。单独走外侧，避免穿过确认节点。
    route([(67.6,51.9),(67.6,44.5),(104.2,44.5),(104.2,42.8)],GRAY,True)
    ax.text(83.8,44.5,"未提名新候选：沿用原版",fontsize=6.35,ha="center",va="center",color=GRAY,
            bbox={"facecolor":"white","edgecolor":"none","pad":1.1},zorder=8)
    route([(102,39.1),(13,39.1),(13,51.5)],TEAL)
    ax.text(55.5,39.1,"下一批从实际生效版本继续；新样本与已返回的答案进入诊断",
            fontsize=6.9,ha="center",va="center",color=TEAL,
            bbox={"facecolor":"white","edgecolor":"none","pad":2},zorder=8)
    ax.plot([1.8,118.5],[32.2,32.2],lw=.6,color=HAIR)
    words(1.8,28.5,"b",11,TEAL,"left",True)
    words(5.7,28.5,"一次真实维护：从乙醛漏判到采用新版",9.5,INK,"left",True)
    cols=[(2.3,27.5),(32.4,56.5),(62.0,86.0),(91.0,117.2)]
    for x0,x1 in cols:
        ax.plot([x0,x1],[23.5,23.5],color="#93AFAE",lw=1)
    words(2.3,20,"首轮探索",7.3,TEAL,"left",True)
    issue = record["issue"]
    words(2.3,14.5,f"{len(split['explore'])} 条探索样本\n{issue['missed']} 条乙醛被判为乙醇",7.1,INK,"left")
    words(32.4,20,"构建定向候选",7.3,TEAL,"left",True)
    words(32.4,14.5,"复查原判乙醇的样本\n原概率与五近邻等权融合",7.1,INK,"left")
    words(62,20,"另一组样本确认",7.3,TEAL,"left",True)
    first,second=adopted["confirmation"]
    words(62,14.5,f"{len(split['adopt'])} 条确认样本 · 宏平均 F1\n{first:.3f} → {second:.3f}",7.1,INK,"left")
    words(91,20,"新版接续后九轮",7.3,TEAL,"left",True)
    later=decisions["later"]
    words(91,14.5,f"{later['count']} 条后续探索样本\n纠正 {later['corrected']} 条，新增错判 {later['new_errors']} 条",
          7.1,INK,"left")
    words(2.3,4.5,"终局开发评价另读已定版模型与留出答案，结果见图 3。",6.9,GRAY,"left")
    return fig


def method_figure(examples, inset):
    fig = plt.figure(figsize=(7.18, 4.05))
    heading(fig,.027,.968,"a","同种响应跨传感器归一化")
    heading(fig,.491,.968,"b","近邻与原模型的等权融合")
    fig.add_artist(Line2D([.466,.466],[.095,.93],transform=fig.transFigure,color=HAIR,lw=.6))
    schematic = fig.add_axes([.04,.512,.39,.39])
    schematic.imshow(plt.imread(inset))
    schematic.axis("off")
    cohort_cmap = LinearSegmentedColormap.from_list("signed-response",["#6A64A2","#F6F7F7","#247976"])
    vmax = max(abs(ex[1]).max() for ex in examples)
    vmax = float(np.ceil(vmax))
    norm = Normalize(-vmax,vmax)
    for i,(sid,shape,_) in enumerate(examples):
        ax = fig.add_axes([.064+i*.201,.162,.121,.272])
        ax.imshow(shape, cmap=cohort_cmap, norm=norm, aspect="auto", interpolation="nearest")
        ax.set_title(f"样本 {sid[-3:]}", size=7.8, pad=4)
        ax.set_xticks([0,7],["1","8"],size=6.5)
        ax.set_yticks([0,15],["1","16"],size=6.5)
        if i == 0:
            ax.set_ylabel("传感器",size=7,labelpad=2)
        ax.tick_params(length=0,pad=2)
        for sp in ax.spines.values(): sp.set_visible(False)
    fig.text(.223,.102,"响应量编号",ha="center",size=7,color=GRAY)
    scale_axis = fig.add_axes([.078,.054,.25,.012])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cohort_cmap),
                 cax=scale_axis,orientation="horizontal",ticks=[-vmax,0,vmax])
    scale_axis.tick_params(labelsize=6,length=0,pad=1)
    scale_axis.set_title("相对响应",fontsize=6.8,pad=2,loc="right")
    # 条宽为同一条样本的六类分数之和。其他四类仅合并显示，胜出判断使用全部六类。
    legend = [Patch(facecolor=c,label=l) for c,l in [(ETHANOL,"乙醇"),(TEAL,"乙醛"),(OTHER,"其余四类")]]
    fig.legend(handles=legend,loc="upper left",bbox_to_anchor=(.497,.909),frameon=False,
               ncol=3,fontsize=7.3,handlelength=1.1,columnspacing=1.2,borderaxespad=0)
    for i,(sid,_,scores) in enumerate(examples):
        y = .808 - i*.389
        truth = "乙醛" if i==0 else "乙醇"
        fig.text(.5,y,f"样本 {sid[-3:]}  ·  真实类别为{truth}",size=8.4,weight="bold")
        ax=fig.add_axes([.61,y-.235,.35,.208])
        ax.set_xlim(0,1); ax.set_ylim(-.4,2.4)
        for j,v in enumerate(scores):
            pos=0.
            for frac,clr in [(v[0],ETHANOL),(v[3],TEAL),(1-v[0]-v[3],OTHER)]:
                ax.barh(2-j,frac,left=pos,height=.58,color=clr,linewidth=0)
                if frac>=.175:
                    ax.text(pos+frac/2,2-j,f"{frac:.1%}".replace(".0%","%"),ha="center",
                            va="center",size=6.6,color="white" if clr!=OTHER else INK)
                pos+=frac
        ax.set_yticks([2,1,0],["原模型","五近邻","等权融合"],size=7.3)
        ax.set_xticks([]); ax.tick_params(length=0,pad=4)
        for sp in ax.spines.values(): sp.set_visible(False)
        outcome="改判为乙醛" if i==0 else "保留乙醇"
        fig.text(.501,y-.284,outcome,fontsize=8,weight="bold",color=TEAL if i==0 else INK)
    return fig


def results_figure(folder):
    summary = read(folder / "comparison.json")
    base = read(folder.parent / "pilot-1/static/result.json")["holdout"]
    if any(v["count"] != base["count"] or abs(v["baseline_macro_f1"]-base["macro_f1"])>1e-10
           for v in summary.values()):
        raise ValueError("方法之间缺少同一开发集和参考分数")
    # 全局读出直接作用于全部样本；局部组报告经采用规则后真正生效的模型。
    rows = [("global-shape-knn","全局近邻",7.55),("global-shape-blend","全局融合",6.65),
            ("original","官方原版",4.88),("pair-prior","训练类频",3.98),
            ("pair-raw-knn","原尺度近邻",3.08),("pair-shape-knn","相对响应近邻",2.18),
            ("pair-shape-blend","相对响应融合",1.28)]
    info = {**summary,"original":{"corrected":0,"new_errors":0,"macro_f1":base["macro_f1"]}}
    fig = plt.figure(figsize=(7.18,3.47))
    heading(fig,.028,.972,"a","相对原版的逐条预测变化")
    heading(fig,.624,.972,"b","六类宏平均 F1")
    left=fig.add_axes([.221,.18,.348,.644])
    right=fig.add_axes([.659,.18,.25,.644])
    left.set(xlim=(-94,48),ylim=(.67,8.65))
    right.set(xlim=(.5,.865),ylim=left.get_ylim())
    left.axvline(0,color="#819298",lw=.65,zorder=0)
    right.axvline(base["macro_f1"],color="#8B989C",lw=.85,linestyle=(0,(2,2)),zorder=0)
    left.text(-.55,8.46,"全局直接读出",transform=left.get_yaxis_transform(),size=7.2,color=GRAY)
    left.text(-.55,5.71,"原版与维护后在用版本",transform=left.get_yaxis_transform(),size=7.2,color=GRAY)
    for key,name,y in rows:
        v=info[key]; added=v["new_errors"]; fixed=v["corrected"]; score=v["macro_f1"]
        bold=key=="pair-shape-blend"
        weight="bold" if bold else "normal"
        left.text(-.035,y,name,transform=left.get_yaxis_transform(),size=7.6,ha="right",
                  va="center",weight=weight,color=TEAL if bold else INK)
        if added: left.barh(y,-added,height=.32,color=RUST)
        if fixed: left.barh(y,fixed,height=.32,color=TEAL)
        if not added and not fixed:
            left.scatter(0,y,s=13,facecolor="white",edgecolor=GRAY,linewidth=.7,zorder=3)
        if added: left.text(-added-2.2,y,str(added),va="center",ha="right",size=7,color=RUST)
        if fixed: left.text(fixed+2.2,y,str(fixed),va="center",size=7,color=TEAL)
        if fixed and not added: left.text(-2.8,y,"0",va="center",ha="right",size=7,color=RUST)
        right.scatter(score,y,s=27 if bold else 20,edgecolor=TEAL if bold else INK,
                      facecolor=TEAL if bold else "white",linewidth=.95,zorder=3)
        right.text(1.03,y,f"{score:.5f}",transform=right.get_yaxis_transform(),size=7.5,
                   va="center",weight=weight,color=TEAL if bold else INK)
    left.set_xticks([-80,-40,0,40])
    left.xaxis.set_major_formatter(FuncFormatter(lambda x,_:str(abs(int(x)))))
    right.set_xticks([.5,.6,.7,.8])
    for ax in (left,right):
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x",labelsize=7,length=2,color=HAIR)
        ax.axhline(6.15,color=HAIR,lw=.5,zorder=0)
    left.set_xlabel("样本数（条）",size=7.4)
    fig.legend(handles=[Patch(facecolor=RUST,label="新增错判"),Patch(facecolor=TEAL,label="纠正漏判或误判")],
               loc="upper left",bbox_to_anchor=(.035,.903),frameon=False,ncol=2,fontsize=7.1,
               handlelength=1.2,columnspacing=1.2,borderaxespad=0)
    fig.text(.642,.864,"虚线为官方原版",size=7,color=GRAY)
    fig.text(.033,.033,f"官方公开开发样本，n = {base['count']}。左图零线上的圆表示纠正和新增错判均为零。",size=7,color=GRAY)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=RUN/"figures")
    parser.add_argument("--font",type=Path,help="可选中文字体文件")
    args=parser.parse_args()
    setup_typeface(args.font)
    figures = [("图1_智训引擎完整流程",overview_figure(RUN)),
               ("图2_定向修正机制",method_figure(example_evidence(RUN),RUN/"figures/assets/同列归一化示意.png")),
               ("图3_修正收益与代价",results_figure(RUN))]
    args.output.mkdir(parents=True,exist_ok=True)
    for name,fig in figures:
        for suffix in ("pdf","png"):
            fig.savefig(args.output/f"{name}.{suffix}",dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
