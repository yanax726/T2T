# Data access and reproduction boundary

## Distributed materials

The repository supplies the current aggregate clinical, molecular, regional and model results, including control analyses, negative findings, uncertainty estimates and sparse or non-estimable cells. It also supplies frozen formula parameters, synthetic calculator examples, binned figure data and manuscript source files. [docs/source_map.md](docs/source_map.md) maps analyses to outputs.

These files support inspection of the reported results and rebuilding their presentation. Aggregates cannot recreate patient-level model training.

## Restricted materials

Patient-level clinical measurements, serum measurements, identifiers, dates, exposure histories, free-text safety records, individual predictions and train/test membership are not included. Their linked clinical content remains restricted. This repository does not grant access to those records or represent that an institutional access process has already been established.

The study team must specify the responsible institution/contact, application criteria, review timeframe and any data-use agreement before offering controlled access. Repository maintainers cannot substitute public GitHub access for that authorization. Do not place individual records in issues or pull requests.

## Authorized refitting

[config/reproduction_inputs.json](config/reproduction_inputs.json) lists each required snapshot, its relative path, availability and SHA-256. With authorized access, restore the exact files at those paths and run:

```bash
PYTHONPATH=src .venv/bin/python -m t2t_paper.preflight
./run.sh analyse
```

The preflight fails if a snapshot is missing or differs. Full refitting begins with these versioned curated snapshots; it does not automatically resolve every discrepancy in the original clinical spreadsheets. It generates new restricted patient-level outputs locally, which must not be committed. In the public repository, the expected preflight failure is evidence of the stated boundary, not a failed aggregate build.

No repository DOI or controlled-access accession has been assigned here. The public Git commit and `SHA256SUMS` identify the distributed snapshot. Any later archive DOI should be added only after an actual deposit.

