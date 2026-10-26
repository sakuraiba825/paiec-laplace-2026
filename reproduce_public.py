"""从公共Parquet独立再生74域×3种子J/A诊断；只保存数值与SHA256。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys

from core import alc_harness_v3_records as runner

CODE = Path(__file__).resolve().parent
BUDGETS = (0, 1, 3, 7, 15, 31)
WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SEEDS = (20261014, 20261015, 20261016)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convert(group):
    items = []
    for item in group.items:
        records = tuple(runner.Record(r.provenance.response_id, r.value, r.interactors)
                        for r in item.responses if r.value is not None)
        if records:
            items.append(runner.Item(item.source_item_id, {"item_content": item.item_content,
                "item_features": item.item_features, "interactors": "", "benchmark_id": item.benchmark_id}, records))
    if not items:
        raise ValueError("完整公共pair没有已评分item")
    return runner.Pair(group.source_benchmark_id + "::" + group.source_subject_id,
                       group.source_benchmark_id, group.source_subject_id, group.subject.as_dict(), tuple(items))


def score_seed(pair, seed, models):
    assumptions = runner.Assumptions(seed, seed, seed, "hash_rank_record")
    split = runner.split_pair(pair, assumptions, below80_policy="exclude_with_report")
    acquisition = set(split.acquisition_keys)
    evaluation = set(split.evaluation_keys)
    stream = [(item, runner.choose_response(item, pair.key, assumptions)) for item in
              runner.order_items(tuple(item for item in pair.items if item.key in acquisition), pair.key, assumptions)]
    eval_records = [(item, record) for item in pair.items if item.key in evaluation for record in item.responses]
    if len(stream) < 31 or not eval_records:
        raise ValueError("冻结pair无法执行31次采集或没有评估response")
    evidence, purchased, cursor = [], [], 0
    budget_briers = {name: [] for name in models}
    checkpoint_hashes = []
    for cap in BUDGETS:
        while len(purchased) < cap and cursor < len(stream):
            item, response = stream[cursor]
            visible = runner.visible_input(pair, item, response)
            context = {
                "subject_id": "local_subject_" + runner._digest(pair.subject_key)[:16],
                "benchmark_id": visible[1]["benchmark_id"],
                "labels_remaining": 31 - len(purchased), "labels_acquired": len(purchased),
                "max_labels": 31, "items_remaining": len(stream) - cursor,
            }
            if runner.default_query(visible, context):
                evidence.append([visible, response.outcome])
                purchased.append([item.key, response.key, response.outcome])
            cursor += 1
        if len(purchased) != cap:
            raise ValueError("冻结公共采集没有达到预算")
        frozen = copy.deepcopy(evidence)
        for name, source in models.items():
            predict = source.load_predict()
            cache = {}
            def losses():
                for item, record in eval_records:
                    visible = runner.visible_input(pair, item, record)
                    key = runner._json(visible)
                    if key not in cache:
                        cache[key] = runner._probability(predict(copy.deepcopy(visible), copy.deepcopy(frozen)))
                    yield (record.outcome - cache[key]) ** 2
            budget_briers[name].append(math.fsum(losses()) / len(eval_records))
        checkpoint_hashes.append({"budget": cap, "purchased_record_sha256": runner._digest(purchased),
                                  "evidence_sha256": runner._digest(frozen)})
    j_alc = math.fsum(w * b for w, b in zip(WEIGHTS, budget_briers["jeffreys"]))
    a_alc = math.fsum(w * b for w, b in zip(WEIGHTS, budget_briers["laplace"]))
    return {
        "seed": seed, "jeffreys_alc": j_alc, "laplace_alc": a_alc,
        "jeffreys_minus_laplace": j_alc - a_alc,
        "jeffreys_budget_brier": budget_briers["jeffreys"],
        "laplace_budget_brier": budget_briers["laplace"], "evaluation_count": len(eval_records),
        "hashes": {"split_fingerprint": split.fingerprint, "split_snapshot_sha256": split.snapshot_sha256,
                   "ordered_candidate_record_ids_sha256": runner._digest([[i.key, r.key] for i, r in stream]),
                   "evaluation_record_ids_labels_sha256": runner._digest([[i.key, r.key, r.outcome] for i, r in eval_records]),
                   "checkpoints": checkpoint_hashes},
    }


def summarize(rows):
    deltas = [row["jeffreys_minus_laplace"] for row in rows]
    rng = random.Random(20261019)
    bootstrap = sorted(statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(10000))
    def budget(index, name):
        return statistics.mean(statistics.mean(seed[name][index] for seed in row["seeds"]) for row in rows)
    return {
        "domain_count": len(rows), "seed_count": sum(len(row["seeds"]) for row in rows),
        "domain_macro_jeffreys_alc": statistics.mean(row["jeffreys_alc"] for row in rows),
        "domain_macro_laplace_alc": statistics.mean(row["laplace_alc"] for row in rows),
        "domain_macro_jeffreys_minus_laplace": statistics.mean(deltas),
        "improved_domains_jeffreys": sum(x < 0 for x in deltas),
        "worst_domain_regression_jeffreys": max(deltas),
        "domain_bootstrap_95_interval": [bootstrap[249], bootstrap[9749]],
        "budget_macro_jeffreys": {str(n): budget(i, "jeffreys_budget_brier") for i, n in enumerate(BUDGETS)},
        "budget_macro_laplace": {str(n): budget(i, "laplace_budget_brier") for i, n in enumerate(BUDGETS)},
        "domain_results": rows,
    }


def compare_reference(result, reference):
    differences = []
    def compare(actual, expected, location):
        if isinstance(expected, dict):
            for key, value in expected.items():
                if key not in ("source_public_summary_sha256", "scope"):
                    compare(actual[key], value, location + "/" + key)
        elif isinstance(expected, list):
            if len(actual) != len(expected):
                raise ValueError("公共参考长度不匹配：" + location)
            for index, value in enumerate(expected):
                compare(actual[index], value, location + "/" + str(index))
        elif isinstance(expected, (int, float)):
            delta = abs(actual - expected)
            differences.append(delta)
            if not math.isfinite(delta) or delta > 1e-12:
                raise ValueError("公共J/A数值对拍失败：" + location)
        elif actual != expected:
            raise ValueError("公共参考标识不匹配：" + location)
    compare(result, reference, "summary")
    return {"passed": True, "absolute_tolerance": 1e-12, "max_absolute_difference": max(differences),
            "checked_numeric_values": len(differences), "reference_sha256": sha256(CODE / "public_reference_summary.json")}


def verify_code_integrity():
    integrity = json.loads((CODE / "release_integrity.json").read_text(encoding="utf-8"))
    for relative, expected in integrity["files_sha256"].items():
        path = CODE / relative
        if not path.resolve().is_relative_to(CODE) or sha256(path) != expected:
            raise ValueError("发布代码或冻结元数据完整性校验失败：" + relative)


def main():
    parser = argparse.ArgumentParser(description="公共74域三种子独立复现；不会读取旧实验或私测文件")
    parser.add_argument("--data-dir", type=Path, required=True, help="公开快照根目录，直接包含域/formatted_tables")
    parser.add_argument("--output-dir", type=Path, required=True, help="新建输出目录；结果不含原题、原始标签或采集轨迹")
    args = parser.parse_args()
    verify_code_integrity()
    from public_snapshot import PublicSnapshot
    manifest = json.loads((CODE / "public_manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["seeds"]) != SEEDS or tuple(runner.BUDGETS) != BUDGETS or tuple(runner.ALC_WEIGHTS) != WEIGHTS:
        raise ValueError("冻结seed/预算/ALC权重不匹配")
    snapshot = PublicSnapshot(args.data_dir, manifest)
    models = {"laplace": runner.ModelSource((CODE / "model.py").read_text(encoding="utf-8"), "model.py"),
              "jeffreys": runner.ModelSource((CODE / "jeffreys_reference.py").read_text(encoding="utf-8"), "jeffreys_reference.py")}
    rows = []
    for entry in manifest["pairs"]:
        pair = convert(snapshot.load_group(entry))
        seeds = [score_seed(pair, seed, models) for seed in SEEDS]
        rows.append({"domain": entry["domain"],
                     "jeffreys_alc": statistics.mean(seed["jeffreys_alc"] for seed in seeds),
                     "laplace_alc": statistics.mean(seed["laplace_alc"] for seed in seeds),
                     "jeffreys_minus_laplace": statistics.mean(seed["jeffreys_minus_laplace"] for seed in seeds),
                     "seeds": seeds})
        print(json.dumps({"completed_domain_count": len(rows), "domain_count": 74,
                          "laplace_alc": rows[-1]["laplace_alc"], "jeffreys_alc": rows[-1]["jeffreys_alc"]}), flush=True)
    result = summarize(rows)
    reference = json.loads((CODE / "public_reference_summary.json").read_text(encoding="utf-8"))
    result["reference_comparison"] = compare_reference(result, reference)
    result.update({
        "status": "complete_independent_public_reproduction",
        "protocol_id": runner.PROTOCOL_VERSION, "budgets": list(BUDGETS), "alc_weights": list(WEIGHTS),
        "verified_public_table_count": len(snapshot.verified),
        "environment": {"python": sys.version.split()[0], "polars": importlib.metadata.version("polars"),
                        "pyarrow": importlib.metadata.version("pyarrow")},
        "source_sha256": {"model.py": sha256(CODE / "model.py"),
                          "jeffreys_reference.py": sha256(CODE / "jeffreys_reference.py"),
                          "public_manifest.json": sha256(CODE / "public_manifest.json"),
                          "release_integrity.json": sha256(CODE / "release_integrity.json")},
        "limitations": [
            "74域是已查看的公共开发域；此诊断不用于新盲测或超参数优化。",
            "split、NULL过滤、record选择及采集顺序是明确本地模拟假设；不是官方服务端或私测成绩复现。",
            "三种子不是222个独立域；bootstrap按74域聚类，仅描述当前域。",
            "每预算重新加载冻结实际predict，对每个不同完整可见input独立预测；每response等权。",
            "不需要旧fold、purchased_records、私测反馈、runtime_config或凭据；不输出原题、原始标签或采集轨迹。",
        ],
    })
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "public_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "domain_count", "seed_count",
          "domain_macro_laplace_alc", "domain_macro_jeffreys_alc", "reference_comparison")}), flush=True)


if __name__ == "__main__":
    main()
