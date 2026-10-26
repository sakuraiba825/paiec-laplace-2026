"""公共新入口的纯合成再生与输出边界测试。"""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from core import alc_harness_v3_records as runner
from reproduce_public import convert, score_seed, compare_reference
from test_protocol import synthetic_pair


class ReproductionTests(unittest.TestCase):
    def test_convert_keeps_raw_record_keys_and_drops_null(self):
        records = [SimpleNamespace(value=value, interactors="synthetic", provenance=SimpleNamespace(response_id=key))
                   for key, value in (("synthetic_record", 1), ("synthetic_null", None))]
        item = SimpleNamespace(source_item_id="synthetic_item", item_content="synthetic_question",
                               item_features="{}", benchmark_id="synthetic_alias", responses=records)
        group = SimpleNamespace(source_benchmark_id="synthetic_benchmark", source_subject_id="synthetic_subject",
                items=[item], subject=SimpleNamespace(as_dict=lambda: {f: "" for f in runner.SUBJECT_FIELDS}))
        pair = convert(group)
        self.assertEqual(pair.key, "synthetic_benchmark::synthetic_subject")
        self.assertEqual([r.key for r in pair.items[0].responses], ["synthetic_record"])

    def test_replay_uses_actual_frozen_model_and_no_raw_outputs(self):
        models = {"laplace": runner.ModelSource((CODE / "model.py").read_text(encoding="utf-8"), "model.py"),
                  "jeffreys": runner.ModelSource((CODE / "jeffreys_reference.py").read_text(encoding="utf-8"), "jeffreys_reference.py")}
        pair = synthetic_pair()
        result = score_seed(pair, 20261014, models)
        self.assertEqual(result, score_seed(pair, 20261014, models))
        self.assertEqual(result["evaluation_count"], 80)
        self.assertEqual(result["laplace_budget_brier"][0], 0.25)
        self.assertEqual(result["jeffreys_budget_brier"][0], 0.25)
        def check(value):
            if isinstance(value, dict):
                forbidden = {"outcome", "response_id", "subject_id", "item_content", "evidence", "purchased_records"}
                self.assertFalse(forbidden & set(value))
                for item in value.values():
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
        check(result)

    def test_prediction_interface_receives_only_twelve_visible_fields(self):
        source = '''
def predict(input, labeled):
    assert len(input) == 2
    assert len(input[0]) == 8 and len(input[1]) == 4
    assert "subject_id" not in input[0] and "response_id" not in input[1]
    assert len(labeled) in (0, 1, 3, 7, 15, 31)
    assert all(entry[1] in (0, 1) for entry in labeled)
    return 0.5
'''
        models = {name: runner.ModelSource(source, "synthetic.py") for name in ("laplace", "jeffreys")}
        result = score_seed(synthetic_pair(), 20261015, models)
        self.assertEqual(result["laplace_budget_brier"], [0.25] * 6)
        self.assertEqual(result["jeffreys_budget_brier"], [0.25] * 6)

    def test_reference_tolerance_is_enforced(self):
        actual = {"value": 0.2, "counts": [74, 222]}
        reference = copy.deepcopy(actual)
        reference["value"] += 1e-13
        self.assertTrue(compare_reference(actual, reference)["passed"])
        reference["value"] += 1e-10
        with self.assertRaises(ValueError):
            compare_reference(actual, reference)


if __name__ == "__main__":
    unittest.main()
