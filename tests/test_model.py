"""纯合成模型接口测试；不读取任何数据集、实验日志或反馈。"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
import model
import jeffreys_reference


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.subject = {name: f"synthetic_{name}" for name in model.SUBJECT_FIELDS}
        self.item = {"benchmark_id": "synthetic_benchmark", "item_content": "synthetic_question",
                     "item_features": "{}", "interactors": ""}
        self.target = [self.subject, self.item]

    def entry(self, label, subject=None, item=None):
        return [[self.subject if subject is None else subject, self.item if item is None else item], label]

    def test_no_labels(self):
        self.assertEqual(model.predict(self.target, []), 0.5)
        self.assertEqual(model.predict(self.target, None), 0.5)

    def test_laplace_formula(self):
        for n in (1, 2, 3, 7, 15, 31, 64):
            for s in range(n + 1):
                with self.subTest(n=n, s=s):
                    labels = [self.entry(1)] * s + [self.entry(0)] * (n - s)
                    self.assertEqual(model.predict(self.target, labels), (1 + s) / (2 + n))

    def test_jeffreys_formula(self):
        for n in (0, 1, 3, 7, 15, 31):
            for s in range(n + 1):
                labels = [self.entry(1)] * s + [self.entry(0)] * (n - s)
                self.assertEqual(jeffreys_reference.predict(self.target, labels), (0.5 + s) / (1 + n))

    def test_all_eight_subject_fields_are_required_to_match(self):
        for name in model.SUBJECT_FIELDS:
            other = {**self.subject, name: "different"}
            with self.subTest(field=name):
                self.assertEqual(model.predict(self.target, [self.entry(1, other)]), 0.5)

    def test_other_benchmark_is_ignored(self):
        self.assertEqual(model.predict(self.target, [self.entry(1, item={**self.item, "benchmark_id": "other"})]), 0.5)

    def test_item_fields_do_not_select_labels(self):
        item = {**self.item, "item_content": "other", "item_features": "other", "interactors": "other"}
        self.assertEqual(model.predict(self.target, [self.entry(1, item=item)]), 2 / 3)

    def test_missing_and_none_subject_fields_normalize_to_empty(self):
        self.assertEqual(model.predict([{}, self.item], [[[dict.fromkeys(model.SUBJECT_FIELDS), self.item], 1]]), 2 / 3)

    def test_supported_binary_label_types(self):
        for label in (True, 1, 1.0, "1", " 1 "):
            self.assertEqual(model.predict(self.target, [self.entry(label)]), 2 / 3)
        for label in (False, 0, 0.0, "0", " 0 "):
            self.assertEqual(model.predict(self.target, [self.entry(label)]), 1 / 3)

    def test_invalid_labels_are_ignored(self):
        for label in (None, -1, 2, 0.5, float("nan"), float("inf"), "1.0", "yes", [], {}):
            self.assertEqual(model.predict(self.target, [self.entry(label)]), 0.5)

    def test_malformed_entries_are_ignored(self):
        malformed = [None, {}, [], [self.target], [None, 1], [[{}], 1], [1, 1], [self.target, 1, 0]]
        self.assertEqual(model.predict(self.target, malformed + [self.entry(1)]), 2 / 3)

    def test_malformed_targets_fallback(self):
        for target in (None, {}, [], [self.subject], [self.subject, self.item, {}], [None, self.item]):
            self.assertEqual(model.predict(target, [self.entry(1)]), 0.5)

    def test_empty_target_benchmark_preserves_frozen_behavior(self):
        self.assertEqual(model.predict([self.subject, {}], [self.entry(1)]), 2 / 3)

    def test_tuple_interface(self):
        self.assertEqual(model.predict(tuple(self.target), [(tuple(self.target), 1)]), 2 / 3)

    def test_input_is_not_mutated(self):
        labels = [self.entry(1)]
        before = copy.deepcopy((self.target, labels))
        model.predict(self.target, labels)
        self.assertEqual((self.target, labels), before)

    def test_extra_ids_do_not_affect_prediction(self):
        target = [{**self.subject, "subject_id": "synthetic_other"}, {**self.item, "item_id": "synthetic_other"}]
        self.assertEqual(model.predict(target, [self.entry(1)]), 2 / 3)

    def test_frozen_model_integrity(self):
        integrity_path = CODE / "release_integrity.json"
        if not integrity_path.exists():
            self.skipTest("发布完整性清单尚未生成")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((CODE / "model.py").read_bytes()).hexdigest(),
                         integrity["files_sha256"]["model.py"])


if __name__ == "__main__":
    unittest.main()
