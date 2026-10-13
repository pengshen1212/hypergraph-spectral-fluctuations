"""One prospective N4-R confirmation; original N4 artifacts stay immutable."""

import argparse
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy

from exact_certificate import exact_cantelli_lower
from finite_sample import tolerance_order_statistic_rank
from n4_runner import (_native_cell_inputs, _interval_for_draw, count_envelope,
                       exact_binomial_interval, order_statistic_threshold,
                       primary_task_consequence_gate)
from spectral_supplement import (SOURCE_PROTOCOL, DIRECTIONS, RESULTS,
                                load_protocol as verify_supplement, read, save,
                                sha256, fraction_text)

HERE = Path(__file__).resolve().parent
NOTES = HERE.parents[2] / 'notes' / 'native-risk'
PROTOCOL = NOTES / 'n4r_protocol.json'
CALIBRATION = NOTES / 'n4r_calibration.json'
CERTIFICATE = NOTES / 'n4r_precommit_certificate.json'
EVALUATION = NOTES / 'n4r_evaluation.json'
SEED_KEYS = ('B_calibration_seed', 'A_batch_1_seed', 'A_batch_2_seed',
             'B_batch_1_seed', 'B_batch_2_seed')
CONFIDENCE = 1 - .05 / 8


def confirmation_cells(original):
    return [dict({k: cell[k] for k in ('id', 'graph', 'profile', 'epsilon')},
                 **{key: 12410001 + i * 100 + j * 100000
                    for j, key in enumerate(SEED_KEYS)})
            for i, cell in enumerate(original['native_cells'])]


def validate_confirmation_cells(cells, original):
    shape = lambda cs: [{k: c[k] for k in ('id', 'graph', 'profile', 'epsilon')} for c in cs]
    if shape(cells) != shape(original['native_cells']):
        raise ValueError('all original cells and profiles must be preserved')
    previous = {c[k] for c in original['native_cells'] for k in SEED_KEYS}
    previous.update(c[k] for c in original['controlled_boundary']['cells']
                    for k in ('B_calibration_seed', 'A_seed', 'B_validation_seed'))
    seeds = [c[k] for c in cells for k in SEED_KEYS]
    if len(seeds) != 40 or len(set(seeds)) != 40 or previous.intersection(seeds):
        raise ValueError('forty distinct fresh streams required')


class NodeSampler:
    def __init__(self, edges, probabilities):
        self.cumulative = np.cumsum(probabilities, axis=1)
        self.classes = probabilities.shape[1]
        self.nodes = np.array([u for e in edges for u in e], dtype=np.intp)
        self.starts = np.r_[0, np.cumsum([len(e) for e in edges])[:-1]]

    def draw(self, rng):
        uniform = rng.random(len(self.cumulative))
        labels = np.minimum((uniform[:, None] >= self.cumulative).sum(axis=1), self.classes - 1)
        slots = labels[self.nodes]
        return (np.minimum.reduceat(slots, self.starts) ==
                np.maximum.reduceat(slots, self.starts)).astype(np.int8)


def recertify_group(group, threshold):
    total, mass, states = Fraction(0), Fraction(0), []
    for state in group['states']:
        probability = Fraction(state['probability_exact'])
        lower = exact_cantelli_lower(Fraction(state['mean_exact']),
                                    Fraction(state['variance_exact']),
                                    Fraction(state['norm_squared_exact']),
                                    Fraction.from_float(float(threshold)))
        total += probability * lower
        mass += probability
        states.append(dict(state=state['state'], probability_exact=state['probability_exact'],
                           lower_exact=fraction_text(lower), lower=float(lower)))
    if mass != 1:
        raise ValueError('every positive-probability state must be retained')
    return dict(nodes=group['nodes'], lower_exact=fraction_text(total), lower=float(total), states=states)


def pool(summaries):
    n = sum(s['draws'] for s in summaries)
    a = sum(s['definite_alarm'] for s in summaries)
    f = sum(s['ambiguous'] for s in summaries)
    return dict(draws=n, definite_alarm=a, ambiguous=f, rate_envelope=[a / n, (a + f) / n],
                simultaneous_adjusted_ci_envelope=[exact_binomial_interval(a, n, CONFIDENCE)[0],
                                                   exact_binomial_interval(a + f, n, CONFIDENCE)[1]])


def summarize(records, threshold):
    result = count_envelope([(r['lo'], r['hi']) for r in records], threshold)
    result.update(pool([result]))
    result['descriptive_95pct_ci_envelope'] = [
        exact_binomial_interval(result['definite_alarm'], result['draws'], .95)[0],
        exact_binomial_interval(result['definite_alarm'] + result['ambiguous'], result['draws'], .95)[1]]
    result['failed_draw_indices'] = [i for i, r in enumerate(records) if 'failure' in r]
    return result


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError('do not overwrite the registered confirmation')
    verify_supplement()
    original = read(SOURCE_PROTOCOL)
    cells = confirmation_cells(original)
    validate_confirmation_cells(cells, original)
    sources = [Path(__file__), HERE / 'test_n4_repair.py', DIRECTIONS, RESULTS,
               HERE / 'spectral_supplement.py', SOURCE_PROTOCOL]
    save(PROTOCOL, dict(id='paper1-native-risk-n4r-confirmation-20261012-v1',
        registered_utc=datetime.now(timezone.utc).isoformat(),
        status='frozen before any N4-R calibration or validation draws',
        inferential_scope='prospective confirmation after retrospective witness development; controlled profiles on two fixed native graphs only',
        cells=cells, calibration_draws=2000, validation_draws_per_batch=1000,
        alpha='1/20', calibration_delta_per_cell='1/800', calibration_rank=1929,
        groups='all original graph-selected singleton and pair groups; all stored states and supported directions without reselection',
        exact_moments='reuse all exact rational conditional moments from the hash-frozen spectral supplement; recompute bounds at fresh thresholds',
        gates=original['frozen_decisions'],
        stop='one confirmation only; no retuning or subsequent confirmatory round regardless of outcome; no threshold-correction method',
        secondary_controls='original completed N4 controls remain historical; not rerun and not part of this confirmation',
        numerical='unchanged verified_norm and same-draw retry rule; keep every failed/ambiguous draw',
        source_hashes={str(p.relative_to(HERE.parents[2])): sha256(p) for p in sources},
        environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
        workers=2, total_draws=48000))
    print('FROZEN N4-R:', sha256(PROTOCOL), flush=True)


def load():
    protocol = read(PROTOCOL)
    verify_supplement()
    for name, digest in protocol['source_hashes'].items():
        if sha256(HERE.parents[2] / name) != digest:
            raise RuntimeError('frozen confirmation input changed: ' + name)
    validate_confirmation_cells(protocol['cells'], read(SOURCE_PROTOCOL))
    return protocol


def stream(cell, arm, seed, count):
    protocol = load()
    path = NOTES / f"n4r_raw_{cell['id']}_{arm}_{seed}.json"
    records = []
    if path.exists():
        previous = read(path)
        if previous['protocol_sha256'] != sha256(PROTOCOL) or previous['count'] != count:
            raise RuntimeError('stream checkpoint does not match protocol')
        records = previous['records']
        if len(records) == count:
            return records
    edges, _, probabilities, pe, builder = _native_cell_inputs(cell)
    sampler = NodeSampler(edges, probabilities)
    rng = np.random.default_rng(seed)
    # Advance the same RNG stream on restart; retained draws are never replaced.
    for _ in records:
        if arm == 'A':
            sampler.draw(rng)
        else:
            rng.random(len(edges))
    for i in range(len(records), count):
        indicators = sampler.draw(rng) if arm == 'A' else (rng.random(len(edges)) < pe).astype(np.int8)
        records.append(_interval_for_draw(builder, indicators))
        if (i + 1) % 100 == 0 or i + 1 == count:
            save(path, dict(protocol_sha256=sha256(PROTOCOL), arm=arm, seed=seed,
                            count=count, records=records))
            print(f"{cell['id']} {arm} seed={seed}: {i+1}/{count}", flush=True)
    return records


def calibrate_cell(cell):
    protocol = load()
    records = stream(cell, 'B', cell['B_calibration_seed'], protocol['calibration_draws'])
    tau = order_statistic_threshold([r['hi'] for r in records], protocol['calibration_rank'])
    return dict(id=cell['id'], threshold=tau, rank=protocol['calibration_rank'],
                failed_draws=[i for i, r in enumerate(records) if 'failure' in r])


def evaluate_cell(cell):
    protocol = load()
    certificate = read(CERTIFICATE)
    if certificate['calibration_sha256'] != sha256(CALIBRATION):
        raise RuntimeError('calibration changed after precommit')
    if certificate['protocol_sha256'] != sha256(PROTOCOL):
        raise RuntimeError('certificate protocol mismatch')
    cert = next(c for c in certificate['cells'] if c['id'] == cell['id'])
    batches = {}
    for key in SEED_KEYS[1:]:
        records = stream(cell, key[0], cell[key], protocol['validation_draws_per_batch'])
        batches[key] = summarize(records, cert['threshold'])
    return dict(id=cell['id'], threshold=cert['threshold'], batches=batches,
                A_pooled=pool([batches['A_batch_1_seed'], batches['A_batch_2_seed']]),
                B_pooled=pool([batches['B_batch_1_seed'], batches['B_batch_2_seed']]),
                precommitted_L=cert['best_lower'])


def run(stage):
    protocol = load()
    target = CALIBRATION if stage == 'calibrate' else EVALUATION
    if target.exists() and read(target).get('complete'):
        raise FileExistsError('completed stage is immutable')
    if stage == 'calibrate':
        rank = tolerance_order_statistic_rank(2000, Fraction(1, 20), Fraction(1, 800))
        if rank != protocol['calibration_rank']:
            raise RuntimeError('calibration rank mismatch')
    elif not CERTIFICATE.exists():
        raise RuntimeError('all certificates must be saved before validation sampling')
    started = time.perf_counter()
    output = dict(protocol_sha256=sha256(PROTOCOL), stage=stage, cells=[], complete=False)
    if stage == 'evaluate':
        output['certificate_sha256'] = sha256(CERTIFICATE)
    with ProcessPoolExecutor(max_workers=protocol['workers']) as executor:
        for result in executor.map(calibrate_cell if stage == 'calibrate' else evaluate_cell, protocol['cells']):
            output['cells'].append(result)
            save(target, output)
            print(f"SAVED {stage} {result['id']}", flush=True)
    output['complete'] = True
    output['wall_seconds'] = time.perf_counter() - started
    if stage == 'evaluate':
        cert = read(CERTIFICATE)
        output['certificate_gate_met'] = cert['primary_certificate_gate_met']
        output['multiclass_nonempty_certificate_gate_met'] = cert['multiclass_nonempty_certificate_gate_met']
        output['task_consequence_gate_met'] = any(primary_task_consequence_gate(
            [row['batches']['A_batch_1_seed'], row['batches']['A_batch_2_seed']],
            row['precommitted_L'], row['A_pooled']['simultaneous_adjusted_ci_envelope'][0]) for row in output['cells'])
        output['original_N4_status'] = 'failed; unchanged'
        output['stop_rule'] = 'confirmation complete; no further upgrade or tuning'
    save(target, output)


def certify():
    protocol = load()
    if CERTIFICATE.exists():
        raise FileExistsError('do not overwrite a precommitted certificate')
    calibration = read(CALIBRATION)
    if not calibration['complete'] or calibration['protocol_sha256'] != sha256(PROTOCOL):
        raise RuntimeError('all fresh calibration streams must finish before certification')
    moments = {c['id']: c for c in read(RESULTS)['cells']}
    output = dict(protocol_sha256=sha256(PROTOCOL), calibration_sha256=sha256(CALIBRATION),
                  directions_sha256=sha256(DIRECTIONS), moments_sha256=sha256(RESULTS),
                  committed_utc=datetime.now(timezone.utc).isoformat(), cells=[])
    for cell in calibration['cells']:
        groups = [recertify_group(g, cell['threshold']) for g in moments[cell['id']]['groups']]
        best = max(Fraction(g['lower_exact']) for g in groups)
        output['cells'].append(dict(id=cell['id'], threshold=cell['threshold'], groups=groups,
                                    best_lower=float(best), best_lower_exact=fraction_text(best)))
    output['primary_certificate_gate_met'] = all(any(
        row['id'].startswith(tag) and Fraction(row['best_lower_exact']) >= Fraction(7, 100)
        for row in output['cells']) for tag in ('cora', 'dblp'))
    output['multiclass_nonempty_certificate_gate_met'] = any(
        'multiclass' in row['id'] and Fraction(row['best_lower_exact']) > 0 for row in output['cells'])
    save(CERTIFICATE, output)
    print('PRECOMMITTED fresh-threshold certificates; validation remains unopened', flush=True)


if __name__ == '__main__':
    sys.set_int_max_str_digits(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'calibrate', 'certify', 'evaluate'))
    args = parser.parse_args()
    {'freeze': freeze, 'calibrate': lambda: run('calibrate'),
     'certify': certify, 'evaluate': lambda: run('evaluate')}[args.stage]()
