"""检验靠已测误差挑选跨界对照瓶能否增益，并保留前一阶段最强固定与论文。"""
from score_bottle_alignment import evaluate_candidates, LocalStore
from score_bottle_challenge import DEST as PREVIOUS
from harness.models.bottle_contrast import ACTIVE, FROZEN, PRIMARY, PAPER, ContrastQuery, ContrastDeployment

DEST = PREVIOUS.parent/"bottle-contrast"

if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("成对标气装机实验已完成")
        evaluate_candidates(DEST,ContrastQuery,ContrastDeployment,ACTIVE,FROZEN,PRIMARY,PAPER,PREVIOUS)
