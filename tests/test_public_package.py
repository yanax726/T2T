"""Check publication boundaries and snapshot preservation on synthetic trees."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from t2t_paper import package


class PublicationBoundaryTest(unittest.TestCase):
    def test_csv_rejects_identifiers_and_accepts_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'counts.csv'
            for content in ['Patient_ID,value\nA,1\n', 'group,n\nPT0012,1\n']:
                path.write_text(content)
                self.assertFalse(package.safe_csv(path))
            path.write_text('group,n,estimate\ncontrol,12,0.2\n')
            self.assertTrue(package.safe_csv(path))

    def test_json_rejects_nested_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'example.json'
            path.write_text(json.dumps({'audit': [{'train_ids': ['A']}]}))
            self.assertFalse(package.safe_json(path))
            path.write_text(json.dumps({'n_patients': 12, 'coefficient': 0.2}))
            self.assertTrue(package.safe_json(path))
            schema = package.ROOT / 'results/tables/update_20261017/model/input_schema.json'
            path.write_bytes(schema.read_bytes())
            self.assertTrue(package.safe_json(path))
            changed = json.loads(path.read_text())
            changed['train_ids'] = ['A']
            path.write_text(json.dumps(changed))
            self.assertFalse(package.safe_json(path))

    def test_each_changed_generated_tree_gets_its_own_verified_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'release/t2t-reproducible'
            output.mkdir(parents=True)
            artifact = output / 'aggregate.csv'
            with patch.object(package, 'ROOT', root), patch.object(package, 'PACKAGE', output):
                artifact.write_text('n\n12\n')
                first = package.snapshot_generated_tree()
                self.assertEqual(first, package.snapshot_generated_tree())
                artifact.write_text('n\n13\n')
                second = package.snapshot_generated_tree()
                self.assertNotEqual(first['tree_sha256'], second['tree_sha256'])
                self.assertTrue((root / first['archive']).exists())
                self.assertTrue((root / second['archive']).exists())

    def test_snapshot_refuses_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'release/t2t-reproducible'
            output.mkdir(parents=True)
            outside = root / 'private.txt'
            outside.write_text('private')
            (output / 'linked.txt').symlink_to(outside)
            with patch.object(package, 'ROOT', root), patch.object(package, 'PACKAGE', output):
                with self.assertRaises(RuntimeError):
                    package.snapshot_generated_tree()


if __name__ == '__main__':
    unittest.main()
