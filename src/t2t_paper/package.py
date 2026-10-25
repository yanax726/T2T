"""Assemble the public research repository from explicit, privacy-screened inputs.

This module writes only release/ and reproducibility receipts. It never publishes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'release/t2t-reproducible'
PRIVATE_COLUMNS = {
    'patient_uid', 'patient_id', 'patient_key', 'participant_id', 'record_id',
    'source_record_id', 'subject_id', 'sample_id', 'train_ids', 'test_ids',
    'visit_date', 'baseline_date', 'free_text', 'name', 'phone', 'dob',
}
STUDY_ID = re.compile(r'\bPT\d{4}\b')
# Reviewed metadata-only dictionary; its column names include participant fields,
# but it has no participant rows. A changed file must be reviewed again.
PUBLIC_SCHEMA_SHA256 = {
    'd805da99365ae77a8b0c1aadfb83d28f5a3c364af7115d6dc437401852f4f321',
}
BASE_FILES = [
    'README.md', 'DATA_ACCESS.md', 'NOTICE.md', 'LICENSE_STATUS.md',
    'CITATION.cff', 'CHANGELOG.md', 'VERSION', 'run.sh', '.gitignore',
    'docs/ci/public-checks.yml',
    'requirements.txt', 'requirements.lock.txt',
    'config/manuscript_revision_20261025.json', 'config/palette_v4_registry.json', 'config/paper_style.json', 'config/paper_figure_plan.json', 'config/figure_identity_map.json',
    'config/marker_registry.csv', 'config/reproduction_inputs.json', 'config/clinical_framework.json',
    'config/unified_model_20261017.json', 'config/unified_bsa_three_v1_20261017.json',
    'config/auxiliary_targets_20261017.json',
    'examples/unified_easi_input.csv', 'examples/unified_easi_log2_input.csv',
    'examples/unified_easi_expected.csv', 'examples/auxiliary_targets_input.csv',
    'tests/test_unified_bsa_three.py', 'tests/test_auxiliary_targets_calculator.py',
    'tests/test_rounding_20261017.py', 'tests/test_redesign_evidence_20261018.py',
    'tests/test_public_package.py',
]
ANALYSIS_MODULES = [
    'clinical_evidence_20261017', 'regional_evidence_20261017',
    'verify_clinical_evidence_20261017', 'verify_regional_evidence_20261017',
    'proposal_closure_20261002', 'update_20261002_overlap', 'update_20261002_analysis',
    'unified_model_20261017', 'rounding_20261017',
    'published_clipping_sensitivity_20261017', 'unified_bsa_three_calculator',
    'reference_sensitivity_20261018',
    'auxiliary_targets_20261017', 'auxiliary_targets_calculator_20261017',
]
PAPER_MODULES = [
    '__init__', 'package', 'demo', 'preflight', 'figures', 'palette', 'build',
    'main_redesign_20261018', 'supplement_redesign_20261018',
    'main_revision_20261025', 'supplement_revision_20261025', 'revision_integration_20261025',
    'redesign_evidence_20261018', 'cohort_flow_20261018',
    'model_display_evidence_20261018', 'balance_support_20261018',
    'plot_source_aggregates_20261018',
    'revision_checks_20261018', 'countmaps', 'clinical_framework_20261018', 'clinical_flow_20261025',
]
FINAL_QA_REPORTS = [
    'reports/deck_alignment_20261025/public_verification.json',
    'reports/deck_alignment_20261025/tests.log',
    'reports/deck_alignment_20261025/isolated_rebuild.log',
    'reports/update_20261018/revision2/statistical_checks_manifest.json',
    'reports/figure_quality/verification.json',
    'reports/figure_quality/public_rebuild.json',
    'reports/figure_quality/public_rebuild_logs_v5/demo.log',
    'reports/figure_quality/public_rebuild_logs_v5/test.log',
    'reports/figure_quality/public_rebuild_logs_v5/rebuild.log',
    'reports/figure_quality/public_rebuild_logs_v5/restricted_preflight.log',
]
REPORT_FILES = [
    'clinical/dependency_contract.json', 'clinical/replay_verification.json',
    'clinical/replay_hashes.csv', 'clinical/result_inventory.csv',
    'clinical/clinical_methods_results_en.md', 'clinical/regional_methods_results.md',
    'clinical/regional_replay_verification.json', 'clinical/overall_moment_verification.json',
    'model/methods.md', 'model/isolated_reproduction_check.json',
    'model/published_clipping_note.md', 'model/auxiliary_targets/methods.md',
    'model/auxiliary_targets/results.md', 'model/auxiliary_targets/isolated_reproduction_check.json',
    'logs/full_pipeline_replay.json',
    'redesign_20261018/cohort_flow_methods.md',
    'redesign_20261018/cohort_flow_manifest.json',
    'redesign_20261018/model_display_methods.md',
    'redesign_20261018/model_display_evidence_manifest.json',
    'redesign_20261018/balance_support_manifest.json',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_report_text(text):
    text = text.replace(str(ROOT), '[PROJECT_ROOT]')
    text = re.sub(r'/Users/[^/\s"\x27]+', '[USER_HOME]', text)
    text = re.sub(r'/private/tmp/[^/\s"\x27]+', '[TEMP_ROOT]', text)
    return STUDY_ID.sub('[restricted source case]', text)


def safe_csv(path):
    path = Path(path)
    if any(part.lower() in {'restricted', 'private'} for part in path.parts):
        return False
    if path.name == 'input.csv' or any(x in path.name.lower() for x in
                                     ('private', 'predictions', 'patient_ledger', 'fold_membership')):
        return False
    with path.open(encoding='utf-8-sig', newline='') as handle:
        columns = {value.strip().lower() for value in next(csv.reader(handle), [])}
    return not PRIVATE_COLUMNS.intersection(columns) and not STUDY_ID.search(
        path.read_text(encoding='utf-8-sig'))


def safe_json(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    if STUDY_ID.search(text):
        return False
    if digest(path) in PUBLIC_SCHEMA_SHA256:
        return True
    def sensitive(value):
        if isinstance(value, dict):
            return any(str(key).lower() in PRIVATE_COLUMNS or sensitive(item)
                       for key, item in value.items())
        if isinstance(value, list):
            return any(sensitive(item) for item in value)
        return False
    return not sensitive(json.loads(text))


def collect_files():
    selected = set(BASE_FILES)
    sanitized = set()
    excluded = []
    for name in ANALYSIS_MODULES:
        selected.add(f'src/cytot2t/curation/{name}.py')
    selected.update({
        'src/cytot2t/clinical_alignment/doctor_evidence.py',
        'src/cytot2t/stages/stage_01_data_lock.py',
        'src/cytot2t/nature_qa/audit_panel_alignment_redevelopment.py',
        'src/cytot2t/nature_qa/LICENSE',
    })
    selected.update(f'src/t2t_paper/{name}.py' for name in PAPER_MODULES)
    for folder, suffixes in [('docs', {'.md', '.json'})]:
        selected.update(str(p.relative_to(ROOT)) for p in (ROOT / folder).glob('*')
                        if p.is_file() and p.suffix in suffixes)

    # Preserve every current aggregate, including controls and negative findings.
    for path in sorted((ROOT / 'results/tables/update_20261017').rglob('*')):
        if not path.is_file() or path.suffix not in {'.csv', '.json'}:
            continue
        relative = str(path.relative_to(ROOT))
        safe = safe_csv(path) if path.suffix == '.csv' else safe_json(path)
        if safe:
            selected.add(relative)
        else:
            excluded.append(dict(path=relative, reason='Restricted row structure, name or study identifier'))

    catalogue = json.loads((ROOT / 'manuscript/figure_catalog.json').read_text())
    ids = [row['id'] for row in catalogue]
    if len(ids) != len(set(ids)) or not all(re.fullmatch(r'Figure(?:S)?[1-9]\d*', x) for x in ids):
        raise RuntimeError('Invalid or duplicate current figure identifier')
    selected.add('manuscript/figure_catalog.json')
    selected.add('manuscript/references.bib')
    for stem in ['main', 'appendix', 'figure_book']:
        selected.update(f'manuscript/{stem}.{ext}' for ext in ['tex', 'pdf'])
    for figure_id in ids:
        selected.update(f'manuscript/figures/{figure_id}.{ext}' for ext in ['pdf', 'svg', 'png'])
    for folder in ['sections', 'tables', 'source_data']:
        for path in (ROOT / 'manuscript' / folder).rglob('*'):
            if path.is_file() and path.suffix in {'.tex', '.csv', '.json'}:
                if path.suffix == '.csv' and not safe_csv(path):
                    raise RuntimeError('Restricted data in manuscript source: ' + str(path))
                if path.suffix == '.json' and not safe_json(path):
                    raise RuntimeError('Restricted data in manuscript JSON: ' + str(path))
                selected.add(str(path.relative_to(ROOT)))
    for name in REPORT_FILES:
        relative = 'reports/update_20261017/' + name
        if (ROOT / relative).exists():
            selected.add(relative)
            sanitized.add(relative)
    for relative in FINAL_QA_REPORTS:
        if (ROOT / relative).exists():
            selected.add(relative)
            sanitized.add(relative)
    for row in catalogue:
        for source in row.get('sources', []):
            source = str(Path(source))
            if source not in selected:
                raise RuntimeError(f'Figure {row["id"]} has an undistributed source: {source}')
    for relative in selected:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('Missing or unsafe publication source: ' + relative)
    return sorted(selected), sanitized, excluded, ids


def zip_tree(folder, destination):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as saved:
        for path in sorted(folder.rglob('*')):
            if path.is_symlink():
                raise RuntimeError('Refusing a symlink in generated package: ' + str(path))
            if path.is_file():
                saved.write(path, str(path.relative_to(folder.parent)))


def snapshot_generated_tree():
    expected = ROOT / 'release/t2t-reproducible'
    if PACKAGE.is_symlink() or PACKAGE.resolve() != expected.resolve():
        raise RuntimeError('Refusing to replace a path outside the generated release tree')
    if not PACKAGE.exists():
        return None
    files = []
    for path in sorted(PACKAGE.rglob('*')):
        if path.is_symlink():
            raise RuntimeError('Refusing a symlink in the previous generated package')
        if path.is_file():
            files.append(dict(path=str(path.relative_to(PACKAGE)), sha256=digest(path)))
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    archive = ROOT / 'archive/generated_publication_snapshots'
    archive.mkdir(parents=True, exist_ok=True)
    snapshot = archive / (identity + '.zip')
    receipt = archive / (identity + '.json')
    if not snapshot.exists():
        zip_tree(PACKAGE, snapshot)
    with zipfile.ZipFile(snapshot) as saved:
        if saved.testzip() is not None:
            raise RuntimeError('Previous package snapshot failed ZIP integrity')
        actual = {str(Path(item.filename).relative_to(PACKAGE.name)):
                  hashlib.sha256(saved.read(item)).hexdigest() for item in saved.infolist()}
    if actual != {row['path']: row['sha256'] for row in files}:
        raise RuntimeError('Previous generated package was not completely preserved')
    receipt.write_text(json.dumps(dict(tree_sha256=identity, archive_sha256=digest(snapshot),
                                      files=files), indent=2) + '\n')
    return dict(tree_sha256=identity, archive=str(snapshot.relative_to(ROOT)),
                archive_sha256=digest(snapshot))


def write_manifest(folder):
    entries = [digest(p) + '  ' + str(p.relative_to(folder)) for p in sorted(folder.rglob('*'))
               if p.is_file() and p.name != 'SHA256SUMS']
    (folder / 'SHA256SUMS').write_text('\n'.join(entries) + '\n')


def verify_public_tree():
    problems = []
    for path in PACKAGE.rglob('*'):
        if not path.is_file():
            continue
        relative = str(path.relative_to(PACKAGE))
        if path.suffix in {'.xlsx', '.xls', '.pptx'} or path.is_symlink():
            problems.append(relative)
        if path.suffix == '.csv' and not safe_csv(path):
            problems.append(relative)
        if path.suffix == '.json' and not safe_json(path):
            problems.append(relative)
        if any(x in relative.lower() for x in ('/reviews/', 'teacher_', 'author_completion',
                                               'handoff', 'release_review')):
            problems.append(relative)
    if problems:
        raise RuntimeError('Publication boundary check failed: ' + ', '.join(sorted(set(problems))))


def run():
    selected, sanitized, excluded, ids = collect_files()
    source_hashes = {path: digest(ROOT / path) for path in selected}
    snapshot = snapshot_generated_tree()
    # Only this verified generated copy is replaced. Scientific source files are untouched.
    if PACKAGE.exists():
        shutil.rmtree(PACKAGE)
    PACKAGE.mkdir(parents=True)
    for relative in selected:
        target = PACKAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative in sanitized:
            target.write_text(public_report_text((ROOT / relative).read_text(encoding='utf-8-sig')))
        else:
            shutil.copy2(ROOT / relative, target)
    for folder in ['src/cytot2t', 'src/cytot2t/curation', 'src/cytot2t/clinical_alignment',
                   'src/cytot2t/stages', 'src/cytot2t/nature_qa']:
        (PACKAGE / folder / '__init__.py').write_text('')
    verify_public_tree()
    changed = [path for path, sha in source_hashes.items() if digest(ROOT / path) != sha]
    different = [path for path, sha in source_hashes.items()
                 if path not in sanitized and digest(PACKAGE / path) != sha]
    if changed or different:
        raise RuntimeError(f'Source changed during packaging or copy differs: {changed + different}')
    aggregate_files = [path for path in selected if path.startswith('results/tables/') and path.endswith('.csv')]
    audit = dict(status='PASS', version=(ROOT / 'VERSION').read_text().strip(),
                 figure_ids=ids, main_figures=sum(not x.startswith('FigureS') for x in ids),
                 supplementary_figures=sum(x.startswith('FigureS') for x in ids),
                 aggregate_csv_count=len(aggregate_files), aggregate_csv_byte_identity=True,
                 selected_files=selected, excluded_aggregate_files=excluded,
                 sanitized_reports=sorted(sanitized), source_sha256=source_hashes,
                 excluded_categories=['Participant inputs and outputs', 'Raw spreadsheets',
                    'Individual predictions and fold membership', 'Clinical dates and safety text',
                    'Internal review/teacher/author-administration materials', 'Presentation decks and notes'],
                 previous_generated_package_snapshot=snapshot)
    local_audit = ROOT / 'reports/update_20261018/revision2/package_audit.json'
    local_audit.parent.mkdir(parents=True, exist_ok=True)
    local_audit.write_text(json.dumps(audit, indent=2) + '\n')
    public_audit = {key: value for key, value in audit.items()
                    if key not in {'excluded_aggregate_files', 'previous_generated_package_snapshot'}}
    (PACKAGE / 'docs/publication_manifest.json').write_text(json.dumps(public_audit, indent=2) + '\n')
    write_manifest(PACKAGE)
    zip_tree(PACKAGE, ROOT / 'release/t2t-reproducible.zip')
    with zipfile.ZipFile(ROOT / 'release/t2t-manuscript-latex.zip', 'w', zipfile.ZIP_DEFLATED) as saved:
        for path in sorted((PACKAGE / 'manuscript').rglob('*')):
            if path.is_file() and path.suffix in {'.tex', '.bib', '.json', '.csv', '.pdf', '.svg'}:
                saved.write(path, str(path.relative_to(PACKAGE)))
    print(json.dumps({key: audit[key] for key in ['status', 'version', 'main_figures',
                     'supplementary_figures', 'aggregate_csv_count']}))
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate the source allowlist without writing a package')
    args = parser.parse_args()
    if args.check:
        files, _, excluded, figures = collect_files()
        print(json.dumps(dict(selected_files=len(files), figure_count=len(figures),
                              excluded_aggregate_files=excluded), indent=2))
    else:
        run()
