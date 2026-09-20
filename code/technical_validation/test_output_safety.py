import tempfile
import unittest
from pathlib import Path
from output_safety import RELEASE_ROOT, OUTPUT_ROOT, check_output

class OutputTests(unittest.TestCase):
    def test_default_external(self):
        self.assertFalse(OUTPUT_ROOT.is_relative_to(RELEASE_ROOT))

    def test_release_rejected(self):
        with self.assertRaises(ValueError):check_output(RELEASE_ROOT/'results')

    def test_existing_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):check_output(d)

    def test_input_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):check_output(Path(d)/'out',d)

    def test_quality_paths_are_config_relative(self):
        import importlib.util
        p=Path(__file__).parent/'data_quality/code/analyze_raw.py'
        spec=importlib.util.spec_from_file_location('quality_paths_test',p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        config=module.load_config(p.parents[1]/'config.yaml')
        self.assertEqual(Path(config['paths']['raw_bids_dir']),RELEASE_ROOT)
        self.assertTrue(Path(config['paths']['output_dir']).is_relative_to(OUTPUT_ROOT))

if __name__=='__main__':unittest.main()
