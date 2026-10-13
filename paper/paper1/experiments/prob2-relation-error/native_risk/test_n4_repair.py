import copy
import unittest
from fractions import Fraction

import numpy as np

import n4_repair as repair
from n4_runner import _native_draw


class RepairTests(unittest.TestCase):
    def test_fresh_seeds_and_exact_original_cells(self):
        original = repair.read(repair.SOURCE_PROTOCOL)
        cells = repair.confirmation_cells(original)
        repair.validate_confirmation_cells(cells, original)
        bad = copy.deepcopy(cells)
        bad[0]['A_batch_1_seed'] = original['native_cells'][0]['A_batch_1_seed']
        with self.assertRaises(ValueError):
            repair.validate_confirmation_cells(bad, original)

    def test_vectorized_indicators_preserve_original_rng_and_events(self):
        edges = [(0, 1), (0, 1, 2), (0, 1), (2, 3)]
        p = np.array([[.2, .3, .5], [.1, .8, .1], [.4, .4, .2], [.7, .2, .1]])
        sampler = repair.NodeSampler(edges, p)
        a, b = np.random.default_rng(884), np.random.default_rng(884)
        for _ in range(30):
            np.testing.assert_array_equal(sampler.draw(a), _native_draw(b, 'A', edges, p, None))
        self.assertEqual(a.random(), b.random())

    def test_recertification_uses_exact_moments_and_all_states(self):
        group = {'nodes': [0], 'states': [
            {'state': [0], 'probability_exact': '1/4', 'mean_exact': '3/1',
             'variance_exact': '1/1', 'norm_squared_exact': '1/1'},
            {'state': [1], 'probability_exact': '3/4', 'mean_exact': '0/1',
             'variance_exact': '1/1', 'norm_squared_exact': '1/1'}]}
        result = repair.recertify_group(group, 1.)
        self.assertEqual(Fraction(result['lower_exact']), Fraction(1, 5))
        self.assertEqual(len(result['states']), 2)
        self.assertEqual(Fraction(repair.recertify_group(group, 3.)['lower_exact']), 0)
        bad = copy.deepcopy(group)
        bad['states'][1]['probability_exact'] = '1/2'
        with self.assertRaises(ValueError):
            repair.recertify_group(bad, 1.)

    def test_ambiguous_draws_retained_and_pooled(self):
        records = [{'lo': 2., 'hi': 3.}, {'lo': 0., 'hi': 1.}, {'lo': 1., 'hi': 3.}]
        summary = repair.summarize(records, 1.5)
        self.assertEqual(summary['draws'], 3)
        self.assertEqual(summary['rate_envelope'], [1 / 3, 2 / 3])
        pooled = repair.pool([summary, summary])
        self.assertEqual(pooled['draws'], 6)
        self.assertEqual(pooled['ambiguous'], 2)


if __name__ == '__main__':
    unittest.main()
