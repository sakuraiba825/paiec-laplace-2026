"""从指定公开元数据生成冻结的最小白名单；不复制其他字段。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="仅提取公共pair选择、公开Parquet路径与SHA256和J/A汇总")
    parser.add_argument("--preregistered", type=Path, required=True)
    parser.add_argument("--download-plan", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.preregistered.read_text(encoding="utf-8"))
    plan = json.loads(args.download_plan.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    pairs = [{key: pair[key] for key in ("domain", "subject_id", "distinct_graded_items", "graded_responses")}
             for pair in selection["train_pairs"]]
    domains = {pair["domain"] for pair in pairs}
    if len(pairs) != len(domains) or len(domains) != 74:
        raise ValueError("公共选择必须是74域各一个完整pair")
    if domains != {row["domain"] for row in reference["domain_results"]}:
        raise ValueError("参考汇总与冻结公共选择域不一致")
    aliases = {domain: f"local_benchmark_{index:04d}"
               for index, domain in enumerate(sorted(plan["eligible_domains"]), 1)
               if domain in domains}
    files = [{key: row[key] for key in ("domain", "table", "repo_path", "expected_sha256", "expected_bytes")}
             for row in plan["files"] if row["domain"] in domains]
    write_new(args.output_dir / "public_manifest.json", {
        "revision": plan["revision"], "filter": plan["filter"],
        "seeds": [20261014, 20261015, 20261016],
        "benchmark_aliases": aliases, "pairs": pairs, "files": files,
        "source_sha256": {"public_selection": sha256(args.preregistered),
                          "public_download_plan": sha256(args.download_plan)},
        "scope": "已查看公共开发域的冻结诊断；不是新盲测、私测反事实或选模门槛",
    })
    scalar_keys = ("domain_count", "seed_count", "domain_macro_jeffreys_alc", "domain_macro_laplace_alc",
                   "domain_macro_jeffreys_minus_laplace", "improved_domains_jeffreys",
                   "worst_domain_regression_jeffreys", "domain_bootstrap_95_interval",
                   "budget_macro_jeffreys", "budget_macro_laplace")
    seed_keys = ("seed", "jeffreys_alc", "laplace_alc", "jeffreys_minus_laplace",
                 "jeffreys_budget_brier", "laplace_budget_brier", "evaluation_count")
    rows = [{"domain": row["domain"], **{key: row[key] for key in
             ("jeffreys_alc", "laplace_alc", "jeffreys_minus_laplace")},
             "seeds": [{key: seed[key] for key in seed_keys} for seed in row["seeds"]]}
            for row in reference["domain_results"]]
    write_new(args.output_dir / "public_reference_summary.json", {
        **{key: reference[key] for key in scalar_keys}, "domain_results": rows,
        "source_public_summary_sha256": sha256(args.reference),
        "scope": "仅保留公共J/A数值；删除旧E指标、fold引用、原始题、标签及轨迹",
    })


if __name__ == "__main__":
    main()
