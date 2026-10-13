import unittest
import n4r_execution as execution


class ExecutionTests(unittest.TestCase):
    def test_override_changes_only_worker_count(self):
        original = {'workers': 2, 'cells': [{'id': 'a', 'seed': 44}], 'gate': .07}
        changed = execution.worker_override(original)
        self.assertEqual(changed['workers'], 8)
        self.assertEqual(original['workers'], 2)
        self.assertEqual({k: v for k, v in changed.items() if k != 'workers'},
                         {k: v for k, v in original.items() if k != 'workers'})


if __name__ == '__main__':
    unittest.main()
