"""纯合成原始record协议测试；全部身份、内容和标签均为测试构造。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import alc_harness_v3_records as runner


def synthetic_pair(count=81):
    subject = {name: "synthetic_" + name for name in runner.SUBJECT_FIELDS}
    items = tuple(runner.Item(f"item_{index:04d}", {
        "item_content": "synthetic_question_" + str(index), "item_features": "{}",
        "interactors": "", "benchmark_id": "synthetic_benchmark"}, (
            runner.Record(f"response_{index:04d}_0", index % 2, "synthetic_a"),
            runner.Record(f"response_{index:04d}_1", 1 - index % 2, "synthetic_b")))
        for index in range(count))
    return runner.Pair("synthetic_pair", "synthetic_benchmark_source", "synthetic_subject_source", subject, items)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pair = synthetic_pair()
        self.assumptions = runner.Assumptions(20261014, 20261014, 20261014, "hash_rank_record")

    def test_split_is_disjoint_complete_and_ceil_half(self):
        split = runner.split_pair(self.pair, self.assumptions, below80_policy="exclude_with_report")
        self.assertFalse(set(split.acquisition_keys) & set(split.evaluation_keys))
        self.assertEqual(len(split.acquisition_keys), 41)
        self.assertEqual(len(split.evaluation_keys), 40)
        self.assertEqual(set(split.acquisition_keys) | set(split.evaluation_keys), {i.key for i in self.pair.items})

    def test_split_order_response_are_deterministic(self):
        for _ in range(2):
            self.assertEqual(runner.split_pair(self.pair, self.assumptions, below80_policy="exclude_with_report"),
                             runner.split_pair(self.pair, self.assumptions, below80_policy="exclude_with_report"))
            self.assertEqual(runner.order_items(self.pair.items, self.pair.key, self.assumptions),
                             runner.order_items(tuple(reversed(self.pair.items)), self.pair.key, self.assumptions))

    def test_response_choice_does_not_use_outcome(self):
        item = self.pair.items[0]
        changed = replace(item, responses=tuple(replace(r, outcome=1-r.outcome) for r in item.responses))
        self.assertEqual(runner.choose_response(item, self.pair.key, self.assumptions).key,
                         runner.choose_response(changed, self.pair.key, self.assumptions).key)

    def test_visible_input_excludes_source_ids_and_labels(self):
        item = self.pair.items[0]
        visible = runner.visible_input(self.pair, item, item.responses[0])
        self.assertEqual(set(visible[0]), set(runner.SUBJECT_FIELDS))
        self.assertEqual(set(visible[1]), set(runner.ITEM_FIELDS))
        self.assertEqual(visible[1]["interactors"], "synthetic_a")

    def test_record_brier_weights_every_response(self):
        items = self.pair.items[:2]
        probabilities = {(i.key, r.key): 0.25 for i in items for r in i.responses}
        self.assertEqual(runner.response_brier(items, probabilities), 0.3125)

    def test_null_and_nonbinary_records_are_rejected(self):
        for value in (None, True, "1", 0.5, 2):
            with self.assertRaises(ValueError):
                runner.Record("synthetic", value, "")

    def test_under80_exclusion_is_explicit(self):
        split = runner.split_pair(synthetic_pair(79), self.assumptions, below80_policy="exclude_with_report")
        self.assertEqual(split.acquisition_keys, ())
        self.assertEqual(split.evaluation_keys, ())

    def test_snapshot_changes_cannot_reuse_split(self):
        split = runner.split_pair(self.pair, self.assumptions, below80_policy="exclude_with_report")
        altered = replace(self.pair, items=self.pair.items[:-1])
        with self.assertRaises(NotImplementedError):
            runner.split_pair(altered, self.assumptions, below80_policy="exclude_with_report", previous=split)

    def test_default_query_guarantees_remaining_purchases(self):
        item = self.pair.items[0]
        visible = runner.visible_input(self.pair, item, item.responses[0])
        context = {"subject_id": "synthetic", "benchmark_id": "synthetic_benchmark", "labels_remaining": 5,
                   "labels_acquired": 26, "max_labels": 31, "items_remaining": 5}
        self.assertTrue(runner.default_query(visible, context))


if __name__ == "__main__":
    unittest.main()
