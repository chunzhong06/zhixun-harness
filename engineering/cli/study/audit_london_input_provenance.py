"""只审计源站输入的真实、沿用和填补来历，不读取目标站评分标签。"""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'engineering/src')]
from engineering.cli.workflow import london_field_driver as scene
from harness.integrations.recipe_regressor import code_inputs
from harness.runtime.storage import atomic_json


def summarize(rows, observed, carried, imputed, filled, age):
    rows = np.asarray(rows, dtype=int)
    count = observed[rows].sum(axis=1)
    known_age = age[rows][age[rows] >= 0]
    return {'rows': len(rows), 'observed_channel_counts': {str(i): int(np.count_nonzero(count == i)) for i in range(observed.shape[1]+1)},
            'observed_fraction_by_channel': observed[rows].mean(axis=0).tolist(),
            'carried_fraction_by_channel': carried[rows].mean(axis=0).tolist(),
            'imputed_fraction_by_channel': imputed[rows].mean(axis=0).tolist(),
            'all_filled_channels_identical': int(np.count_nonzero(np.ptp(filled[rows], axis=1) == 0)),
            'no_observation_so_far': int(np.count_nonzero(age[rows] < 0)),
            'any_observation_age_quantiles_hours': np.quantile(known_age, [0, .5, .9, .99, 1]).tolist() if len(known_age) else None}


def run(args):
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    archive = ROOT/'data/public/london-air-2026/stations.zip'
    scene.configure('2026-07-01')
    scene.CLOCK = pd.date_range(scene.START-pd.Timedelta(days=180), scene.START+pd.Timedelta(days=62, hours=-1), freq='h')
    raw = pd.concat([scene.readings(archive, site).rename(site) for site in scene.DONORS], axis=1)
    forward = raw.ffill(limit=6)
    median = forward.median(axis=1).ffill().fillna(0.)
    filled = forward.apply(lambda column: column.fillna(median)).to_numpy()
    features, names = code_inputs(scene.prepare(archive, True, True))
    assert names[:4] == list(scene.DONORS)
    np.testing.assert_array_equal(features[:, :4], filled)
    observed = raw.notna().to_numpy()
    carried = raw.isna().to_numpy() & forward.notna().to_numpy()
    imputed = forward.isna().to_numpy()
    np.testing.assert_array_equal(observed.astype(int)+carried.astype(int)+imputed.astype(int), np.ones_like(filled, dtype=int))
    any_seen = np.maximum.accumulate(np.where(observed.any(axis=1), np.arange(len(raw)), -1))
    age = np.where(any_seen >= 0, np.arange(len(raw))-any_seen, -1)
    start = int(np.flatnonzero(scene.CLOCK >= scene.START)[0]);at = start+28*24
    windows = {'initial_history': np.arange(start-21*24-6, start),
               'maintenance_history': np.arange(at-21*24-6, at),
               'first_28d': np.arange(start, at), 'postpublication_28d': np.arange(at, start+56*24),
               'tail_6d': np.arange(start+56*24, len(raw))}
    summary = {name: summarize(rows, observed, carried, imputed, filled, age) for name, rows in windows.items()}
    daily = []
    for day in range(62):
        rows = np.arange(start+24*day, start+24*(day+1))
        record = {'day': day, 'date': scene.CLOCK[rows[0]].date().isoformat(), **summarize(rows, observed, carried, imputed, filled, age)}
        daily.append(record)
    # 仅源站及生成过程的来历；可作为后续输入契约扩展的原始核对资料。
    np.savez_compressed(output/'provenance.npz', hours=np.arange(len(raw)), filled=filled,
                        observed=observed, carried=carried, imputed=imputed, any_observation_age=age)
    atomic_json(output/'daily.json', daily)
    report = {'scope': '源输入来历审计；0目标站答案、0模型训练或调用', 'donors': list(scene.DONORS),
              'archive_sha256': sha256(archive.read_bytes()).hexdigest(), 'existing_feature_values_exactly_reproduced': True,
              'mutually_exclusive_provenance_verified': True, 'periods': summary}
    atomic_json(output/'audit.json', report);print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = ArgumentParser();p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
