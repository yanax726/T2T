"""Check restricted snapshot availability before a full statistical refit."""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]

def main():
    contract = ROOT/'config/reproduction_inputs.json'
    inputs = json.loads(contract.read_text())['inputs']
    errors = []
    for row in inputs:
        path = ROOT/row['path']
        if not path.exists():
            errors.append('Not supplied: '+row['path'])
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Hash differs: '+row['path'])
    if errors:
        raise SystemExit('\n'.join(errors)+'\nFull analysis requires authorized restricted snapshots. Use ./run.sh rebuild for the aggregate-data reproduction.')
    print(f'PASS: {len(inputs)} declared snapshots match their hashes.')

if __name__ == '__main__':
    main()
