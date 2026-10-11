"""Audit-only contracts: exclusions, unambiguous pairing and mask semantics."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from dermssl import view_audit as audit


class ViewAuditContracts(unittest.TestCase):
    def test_exclusions_and_deterministic_sample(self):
        images = {str(i): Path(str(i)) for i in range(12)}
        selected = audit.select_ids(images, images, images, {"0", "1"}, 5, 42)
        self.assertEqual(selected, audit.select_ids(images, images, images, {"0", "1"}, 5, 42))
        self.assertFalse(set(selected) & {"0", "1"})
        self.assertEqual(len(selected), len(set(selected)))

    def test_missing_pairs_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "Missing transformed"):
            audit.select_ids({"a": Path("a")}, {}, {"a": Path("a")}, set(), 1, 42)

    def test_duplicate_stems_are_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "a.png")
            Image.new("RGB", (8, 8)).save(root / "a.jpg")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                audit.index_images(root)

    def test_ignore_255_is_never_lesion_under_pseudo_encoding(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.array([[0, 1, 255]], dtype=np.uint8)).save(path)
            mask = audit.load_mask(path, (3, 1), "pseudo01255")
            np.testing.assert_array_equal(mask, [[False, True, False]])
            with self.assertRaises(ValueError):
                audit.load_mask(path, (3, 1), "binary255")
            with self.assertRaisesRegex(ValueError, "grids"):
                audit.load_mask(path, (4, 1), "pseudo01255")

    def test_crop_retention_and_empty_mask_have_distinct_meanings(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:4, 2:4] = True
        stats = audit.crop_stats(mask, (0, 0, 3, 3))
        self.assertEqual(stats["lesion_retained_fraction"], 0.25)
        self.assertFalse(stats["lesion_absent"])
        empty = audit.crop_stats(np.zeros_like(mask), (0, 0, 3, 3))
        self.assertTrue(empty["source_mask_empty"])
        self.assertIsNone(empty["lesion_retained_fraction"])


if __name__ == "__main__":
    unittest.main()
