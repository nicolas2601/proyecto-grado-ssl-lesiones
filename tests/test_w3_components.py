import unittest
import numpy as np
from dermssl.w3_components import (DegenerateCase, USLTextPolicy, gaussian_weights,
    sled_location_selection, normalize_sled_scores, fuse_sled_scales, usl_text_pseudolabel)
from dermssl.w3_sled_adapter import run_sled_rgb, sled_parameters


class W3ComponentsTest(unittest.TestCase):
    def test_gaussian_shape_symmetry_and_center(self):
        w = gaussian_weights((60, 80))
        self.assertEqual(w.shape, (60, 80))
        self.assertAlmostEqual(w.sum(), 1)
        np.testing.assert_allclose(w, w[::-1, ::-1])
        self.assertGreater(w[30, 40], w[10, 10])

    def test_correction_changes_selected_component(self):
        candidate = np.zeros((100, 100), bool)
        candidate[45:55, 45:55] = True  # small central component
        candidate[15:35, 20:40] = True  # larger peripheral component
        valid = np.ones_like(candidate)
        old = sled_location_selection(candidate, valid, variant='upstream_area')
        new = sled_location_selection(candidate, valid)
        self.assertTrue(old[25, 30])
        self.assertFalse(old[50, 50])
        self.assertTrue(new[50, 50])
        self.assertFalse(new[25, 30])

    def test_fill_cannot_restore_invalid_pixels(self):
        candidate = np.zeros((50, 50), bool); candidate[15:35, 15:35] = True
        valid = np.ones_like(candidate); valid[23:27, 23:27] = False
        self.assertFalse(sled_location_selection(candidate, valid)[25, 25])
        self.assertTrue(sled_location_selection(candidate, valid, constrain_filled_holes=False)[25, 25])

    def test_sled_degenerate_inputs_fail_explicitly(self):
        with self.assertRaises(DegenerateCase): normalize_sled_scores(np.zeros((3, 3)))
        with self.assertRaises(DegenerateCase): fuse_sled_scales([np.ones((3, 3))], [0])
        with self.assertRaises(DegenerateCase): sled_location_selection(np.zeros((3, 3)), np.ones((3, 3)))

    def test_stable_fusion_preserves_weights_without_overflow(self):
        maps = [np.zeros((2, 2)), np.ones((2, 2))]
        expected = np.exp(1) / (1 + np.exp(1))
        np.testing.assert_allclose(fuse_sled_scales(maps, [1, 2]), expected)
        np.testing.assert_allclose(fuse_sled_scales(maps, [1, 10000]), 1)

    def test_sum_centrality_is_not_centroid_centrality(self):
        s = np.full((224, 224), .2)
        s[70:140, 70:135] = .9  # near-center centroid but 4550 pixels
        s[105:125, 140:170] = .9  # farther centroid, only 600 pixels
        out, _ = usl_text_pseudolabel(s)
        self.assertEqual(out[110, 150], 1)
        self.assertEqual(out[110, 110], .5)
        self.assertEqual(out[0, 0], 0)

    def test_exact_thresholds_are_uncertain_and_small_regions_rejected(self):
        s = np.full((224, 224), .35); s[90:115, 90:115] = .9
        s[0, 0] = .65; s[20:30, 20:30] = .9
        out, _ = usl_text_pseudolabel(s)
        self.assertEqual(out[0, 0], .5)
        self.assertEqual(out[25, 25], .5)
        self.assertEqual(out[200, 200], .5)

    def test_abstention_and_declared_cutoff(self):
        with self.assertRaises(DegenerateCase): usl_text_pseudolabel(np.zeros((224, 224)))
        s = np.zeros((224, 224)); s[20:50, 20:50] = .9
        with self.assertRaises(DegenerateCase):
            usl_text_pseudolabel(s, USLTextPolicy(max_centroid_distance=10))
        for value in [np.nan, -1, 2]:
            s[0, 0] = value
            with self.assertRaises(ValueError): usl_text_pseudolabel(s)

    def test_adapter_geometry_is_checked_before_selection(self):
        class BadBackend:
            def SLED_single_scale(self, *args):
                return np.zeros((2, 2)), None, 1, 'case'
        with self.assertRaises(DegenerateCase):
            run_sled_rgb(np.zeros((30, 30, 3), np.uint8), np.ones((30, 30)),
                         BadBackend(), sled_parameters(), scales=(200,))


if __name__ == '__main__': unittest.main()
