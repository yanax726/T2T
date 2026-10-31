"""Catch colour-semantic failures across panels without interpreting heatmaps as groups."""
from copy import deepcopy
import unittest
import warnings
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgb

from t2t_paper import palette
from t2t_paper import visual_qa_20261031 as vision


class PaletteSemanticTest(unittest.TestCase):
    def setUp(self):
        self.registry = deepcopy(palette.REGISTRY)
        self.registry['figure_rules']['Test'] = {
            'categorical_families': ['treatment', 'visit', 'method', 'recorded_iga_0_to_5'],
            'continuous_scales': ['ocean_coral_diverging', 'ocean_sequential']}
        self.registry_patch = patch.object(palette, 'REGISTRY', self.registry)
        self.registry_patch.start()
        self.fig, self.axes = plt.subplots(1, 2)

    def tearDown(self):
        plt.close(self.fig)
        self.registry_patch.stop()

    def audit(self):
        self.fig.canvas.draw()
        return palette.audit_figure(self.fig, 'Test')

    def test_registered_but_never_drawn_is_not_pass(self):
        palette.register(self.fig, 'visit.W12')
        with self.assertRaisesRegex(ValueError, 'no visible coloured artist'):
            self.audit()

    def test_changed_colour_after_binding_fails(self):
        role = 'treatment.dupilumab'
        line, = self.axes[0].plot([0, 1], [1, 2], color=palette.color(role))
        palette.bind_artist(self.fig, role, line)
        line.set_color(palette.color('treatment.abrocitinib_200mg'))
        with self.assertRaisesRegex(ValueError, 'bound to'):
            self.audit()

    def test_reusing_one_rgb_for_different_roles_across_panels_fails(self):
        self.registry['visit']['W12']['hex'] = palette.color('treatment.dupilumab')
        self.axes[0].plot([0, 1], [1, 2], color=palette.color('treatment.dupilumab'))
        self.axes[1].plot([0, 1], [2, 1], color=palette.color('visit.W12'))
        with self.assertRaisesRegex(ValueError, 'ambiguous categorical roles'):
            self.audit()

    def test_disabled_marker_colour_does_not_invent_a_group(self):
        self.axes[0].plot([0, 1], [1, 2], color='black', marker='None',
                          markerfacecolor=palette.color('visit.W12'))
        report = self.audit()
        self.assertNotIn('visit.W12', report['roles'])

    def test_hidden_and_transparent_artists_do_not_prove_registration(self):
        role = 'visit.W24'
        palette.register(self.fig, role)
        self.axes[0].plot([0, 1], [1, 2], color=palette.color(role), visible=False)
        self.axes[1].plot([0, 1], [1, 2], color=palette.color(role), alpha=0)
        with self.assertRaisesRegex(ValueError, 'no visible coloured artist'):
            self.audit()

    def test_empty_line_and_unfilled_marker_face_do_not_invent_a_group(self):
        c = palette.color('visit.W12')
        self.axes[0].plot([], [], color=c)
        self.axes[1].plot([0, 1], [1, 2], color='black', marker='x',
                          markeredgecolor='black', markerfacecolor=c)
        report = self.audit()
        self.assertNotIn('visit.W12', report['roles'])

    def test_shaded_band_and_open_markers_keep_one_role(self):
        role = 'treatment.abrocitinib_100mg'
        c = palette.register(self.fig, role)
        self.axes[0].plot([0, 1], [1, 2], color=c, marker='o', markerfacecolor='white')
        self.axes[0].fill_between([0, 1], [0, 1], [2, 3], color=c, alpha=.16)
        tint = 1 + .21 * (np.asarray(to_rgb(c)) - 1)
        bars = self.axes[1].bar([0], [1], color=tint)
        palette.bind_artist(self.fig, role, bars)
        report = self.audit()
        self.assertEqual(list(report['roles']), [role])
        self.assertFalse(report['unregistered_chromatic_colours'])
        self.assertIn('derived_fill', report['observed_role_details'][role]['variants'])

    def test_inferred_roles_are_not_misreported_as_explicitly_registered(self):
        role = 'visit.W0'
        self.axes[0].scatter([0, 1], [1, 2], color=palette.color(role))
        report = self.audit()
        self.assertIn(role, report['inferred_roles'])
        self.assertNotIn(role, report['declared_roles'])

    def test_pale_border_does_not_create_an_unrelated_categorical_role(self):
        self.registry['figure_rules']['Test']['categorical_families'] = []
        self.axes[0].bar([0], [1], facecolor='white', edgecolor='#E5D7CE')
        report = self.audit()
        self.assertFalse(report['roles'])

    def test_known_colour_from_wrong_family_is_caught_without_registration(self):
        self.registry['figure_rules']['Test']['categorical_families'] = ['visit']
        self.axes[0].plot([0, 1], [1, 2], color=palette.color('treatment.dupilumab'))
        with self.assertRaisesRegex(ValueError, 'unexpected categorical roles'):
            self.audit()

    def test_heatmap_lut_is_not_interpreted_as_categorical_marks(self):
        # A numerical colour may happen to equal a categorical colour. Its
        # meaning follows the scalar mappable and colour bar, not that HEX.
        self.registry['continuous']['test'] = {
            'lut_hex': [palette.color('treatment.dupilumab'), '#FFFFFF', '#EE4411']}
        self.registry['figure_rules']['Test']['categorical_families'] = []
        self.registry['figure_rules']['Test']['continuous_scales'] = ['test']
        im = self.axes[0].imshow([[0, 1], [1, 2]], cmap=palette.cmap('test'))
        self.fig.colorbar(im, ax=self.axes[0])
        report = self.audit()
        self.assertFalse(report['roles'])
        self.assertFalse(report['warnings'])
        self.assertTrue(all(x['registry_scale'] == 'test' for x in report['observed_continuous_scales']))

    def test_count_map_subset_has_registry_provenance(self):
        lut = self.registry['continuous']['ocean_sequential']['lut_hex'][45:]
        self.axes[0].imshow([[0, 1], [2, 3]], cmap=ListedColormap(lut, name='ocean_patient_counts'))
        report = self.audit()
        self.assertEqual(report['observed_continuous_scales'][0]['registry_scale'], 'ocean_sequential')
        self.assertFalse(report['warnings'])

    def test_unknown_colour_warns_but_neutral_reference_line_does_not(self):
        self.axes[0].plot([0, 1], [1, 2], color='#FF00FF')
        self.axes[1].axhline(0, color='#AAB6BC')
        report = self.audit()
        self.assertEqual(set(report['unregistered_chromatic_colours']), {'#FF00FF'})
        self.assertEqual(report['status'], 'REVIEW')

    def test_errorbar_container_binds_markers_and_interval(self):
        role = 'visit.W12'
        container = self.axes[0].errorbar([0, 1], [1, 2], yerr=.2,
                                          color=palette.color(role), fmt='o', capsize=2)
        palette.bind_artist(self.fig, role, container)
        report = self.audit()
        self.assertEqual({x['role'] for x in report['bound_artists']}, {role})
        self.assertGreaterEqual(len(report['bound_artists']), 4)


class VisionSimulationTest(unittest.TestCase):
    def test_primary_colour_pixels_agree_with_scalar_arithmetic(self):
        pixels = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [.23, .67, .41], [0., 0., 0.], [1., 1., 1.]])
        def decode(x):
            return x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4
        def encode(x):
            x = max(0., min(1., x))
            return 12.92 * x if x <= .0031308 else 1.055 * x ** (1 / 2.4) - .055
        for mode in vision.MODES:
            expected = []
            for pixel in pixels:
                decoded = [decode(float(x)) for x in pixel]
                if mode == 'original':
                    expected.append(pixel.tolist())
                elif mode == 'grayscale':
                    gray = encode(sum(x * weight for x, weight in zip(decoded, [.2126, .7152, .0722])))
                    expected.append([gray] * 3)
                else:
                    expected.append([encode(sum(x * coefficient for x, coefficient in zip(decoded, row)))
                                     for row in vision.MATRICES[mode]])
            np.testing.assert_allclose(vision.simulate(pixels, mode), expected, atol=2e-15)

    def test_large_noncontiguous_image_is_finite_and_nonconstant_without_warnings(self):
        rng = np.random.default_rng(73)
        image = rng.uniform(0, 1, (384, 1024, 3))[:, ::2]
        self.assertFalse(image.flags.c_contiguous)
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            for mode in vision.MODES:
                result = vision.simulate(image, mode)
                self.assertEqual(result.shape, image.shape)
                self.assertTrue(np.isfinite(result).all())
                self.assertGreaterEqual(result.min(), 0.)
                self.assertLessEqual(result.max(), 1.)
                self.assertGreater(float(np.std(result)), .1)
                if mode == 'original':
                    np.testing.assert_array_equal(result, image)
                else:
                    self.assertGreater(np.count_nonzero(np.abs(result - image) > .01), image.size / 2)
                if mode == 'grayscale':
                    np.testing.assert_array_equal(result[..., 0], result[..., 1])
                    np.testing.assert_array_equal(result[..., 1], result[..., 2])

    def test_nonfinite_or_wrongly_scaled_rgb_fails_before_clipping(self):
        for invalid in ([[np.nan, 0., 0.]], [[np.inf, 0., 0.]], [[255., 0., 0.]], [[-.1, 0., 0.]]):
            with self.assertRaises(ValueError):
                vision.simulate(invalid, 'grayscale')
        with self.assertRaises(ValueError):
            vision.srgb([[np.inf, 0., 0.]])


if __name__ == '__main__':
    unittest.main()
