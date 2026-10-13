"""Pre-validation execution-only worker amendment; scientific runner unchanged."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import n4_repair as repair

AMENDMENT = repair.NOTES / 'n4r_execution_workers.json'


def worker_override(protocol):
    return dict(protocol, workers=8)


def freeze():
    if AMENDMENT.exists():
        raise FileExistsError('worker amendment already frozen')
    repair.load()
    if list(repair.NOTES.glob('n4r_raw_*_A_*.json')):
        raise RuntimeError('execution amendment must precede validation')
    repair.save(AMENDMENT, dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=repair.sha256(repair.PROTOCOL),
        launcher_sha256=repair.sha256(Path(__file__)),
        tests_sha256=repair.sha256(Path(__file__).with_name('test_n4r_execution.py')),
        from_workers=2, to_workers=8,
        scope='validation scheduling only; every frozen scientific function and stream unchanged',
        reason='DBLP per-draw sparse norm cost; available independent CPU cores',
        status='frozen before all validation draws'))


def evaluate():
    original_load = repair.load
    protocol = original_load()
    amendment = repair.read(AMENDMENT)
    if (amendment['protocol_sha256'] != repair.sha256(repair.PROTOCOL) or
            amendment['launcher_sha256'] != repair.sha256(Path(__file__)) or
            amendment['tests_sha256'] != repair.sha256(Path(__file__).with_name('test_n4r_execution.py'))):
        raise RuntimeError('worker execution amendment hash mismatch')
    # Override only the in-memory scheduling parameter. The source protocol,
    # runner and every scientific helper remain hash-verified and untouched.
    repair.load = lambda: worker_override(original_load())
    repair.run('evaluate')
    output = repair.read(repair.EVALUATION)
    output['execution_amendment_sha256'] = repair.sha256(AMENDMENT)
    repair.save(repair.EVALUATION, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'evaluate'))
    args = parser.parse_args()
    {'freeze': freeze, 'evaluate': evaluate}[args.stage]()
