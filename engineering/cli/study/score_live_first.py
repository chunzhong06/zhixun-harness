"""将首批现场规则的实际交接与既有同输入对照配对计分，中断后可逐来源重算。"""
from argparse import ArgumentParser

from compare_error_slice import DEST as REPEATED
from compare_proposal_frequency import BASE, OUTPUT, TASKS, atomic_json, measure_streams, read, reference_folders

DEST = BASE / "live-first"


def score(cohort):
    task = TASKS[cohort]
    fixed, paper = reference_folders(cohort, "mature")
    public = task == cohort
    sub = ("public", task) if public else (cohort,)
    paths = {"default": BASE/"selected"/task if public else fixed.parent/"project",
             "fixed": fixed, "mature": paper, "paper-first": OUTPUT/cohort/"mature",
             "error-every": REPEATED.joinpath(*sub), "error-first": DEST.joinpath(*sub)}
    if not public:
        paths["simple"] = fixed.parent/"simple"
    replies = {arm: [] for arm in paths}
    replies["mature"] = read(paper/"driver-exchanges.json")["exchanges"]
    replies["paper-first"] = replies["mature"][:1]
    if not public:
        replies["simple"] = read(paths["simple"]/"driver-exchanges.json")["exchanges"]
    table, streams, control = measure_streams(cohort, paths, replies)
    report = read(paths["error-first"]/"run.json")
    if report["policy"] != "first-proposal" or report["driver"] != "rule":
        raise ValueError("需使用本地规则现场提案的交接记录")
    for arm, stream in streams.items():
        table[arm]["changed_from_fixed"] = sum(stream[k] != v for k, v in streams["fixed"].items())
    atomic_json(DEST/"scores"/(cohort+".json"), {"cohort":cohort, "input_and_budget":control,
                "first_action":report["rounds"][0]["planned_actions"], "arms":table})
    print(cohort, {k:round(v["score"], 5) for k,v in table.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=(*TASKS, "all"))
    args = parser.parse_args()
    for name in TASKS if args.cohort == "all" else [args.cohort]:
        score(name)
