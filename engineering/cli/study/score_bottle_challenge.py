"""复核保留全量程标气后再主动询问与分界校准的增量。"""
from score_bottle_alignment import DEST as PREVIOUS, evaluate_candidates, LocalStore
from harness.models.bottle_challenge import ACTIVE, FROZEN, PRIMARY, PAPER, BalancedQuery, ChallengeDeployment

DEST = PREVIOUS.parent/"bottle-challenge"

if __name__ == "__main__":
    with LocalStore(DEST/"lock").exclusive():
        if (DEST/"scores.json").exists():
            raise ValueError("平衡量程的装机实验已完成")
        evaluate_candidates(DEST,BalancedQuery,ChallengeDeployment,ACTIVE,FROZEN,PRIMARY,PAPER,PREVIOUS)
