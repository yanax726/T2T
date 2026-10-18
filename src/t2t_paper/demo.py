"""Run the frozen EASI calculator on synthetic inputs and check its output."""
from pathlib import Path
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time


def run(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    output = root / 'results/demo'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / 'unified_easi_output.csv'
    command = [sys.executable, '-m', 'cytot2t.curation.unified_bsa_three_calculator',
               str(root / 'examples/unified_easi_input.csv'), str(destination),
               '--root', str(root), '--input-scale', 'raw']
    started = time.perf_counter()
    subprocess.run(command, check=True, cwd=root, env={**os.environ,
        'PYTHONPATH': str(root / 'src') + os.pathsep + os.environ.get('PYTHONPATH', '')})
    elapsed = time.perf_counter() - started
    with destination.open(newline='') as handle:
        actual = list(csv.DictReader(handle))
    with (root / 'examples/unified_easi_expected.csv').open(newline='') as handle:
        expected = list(csv.DictReader(handle))
    if len(actual) != len(expected):
        raise AssertionError('Demo row count differs from the expected output')
    for row_number, (observed, reference) in enumerate(zip(actual, expected), 1):
        if observed.keys() != reference.keys():
            raise AssertionError('Demo output columns differ')
        for field in reference:
            if not math.isclose(float(observed[field]), float(reference[field]),
                                rel_tol=0, abs_tol=1e-12):
                raise AssertionError(f'Demo row {row_number}, {field} differs')
    report = dict(status='PASS', rows=len(actual), absolute_tolerance=1e-12,
                  calculator_seconds=elapsed, python=platform.python_version(),
                  platform=platform.platform(), machine=platform.machine(),
                  data='Synthetic examples, not participant measurements')
    (output / 'runtime.json').write_text(json.dumps(report, indent=2) + '\n')
    print(destination.read_text().rstrip())
    print(f'PASS: {len(actual)} synthetic rows match expected output (absolute tolerance 1e-12).')
    print(f'Calculator runtime: {elapsed:.3f} seconds. Output: results/demo/unified_easi_output.csv')
    return report


if __name__ == '__main__':
    run()
