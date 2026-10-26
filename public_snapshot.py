"""只读冻结白名单的公共Parquet；不读旧实验、反馈或本机运行配置。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import polars as pl
import pyarrow.parquet as pq

from core.measurement_snapshot import (
    REVISION, SCHEMAS, SUBJECT_FIELDS, TABLES, VisibleSubject,
    ResponseProvenance, SnapshotResponse, SnapshotItem, SnapshotGroup,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    return "" if value is None else value


class PublicSnapshot:
    def __init__(self, data_dir, manifest):
        self.root = Path(data_dir).resolve()
        require(manifest["revision"] == REVISION, "公共快照revision不匹配")
        require(manifest["filter"] == {"release": "public", "response_type": "binary", "granularity": "item"},
                "仅允许公共binary item快照")
        self.pairs = tuple(manifest["pairs"])
        domains = {row["domain"] for row in self.pairs}
        require(len(domains) == len(self.pairs) == 74, "冻结选择必须是74域各一个pair")
        require(all(re.fullmatch(r"[a-z][a-z0-9_]*", d) and not d.startswith("private") for d in domains),
                "白名单包含非法域")
        self.aliases = manifest["benchmark_aliases"]
        require(set(self.aliases) == domains and len(set(self.aliases.values())) == 74,
                "benchmark alias不是一一对应")
        self.files = {(row["domain"], row["table"]): row for row in manifest["files"]}
        require(len(self.files) == len(manifest["files"]) and
                set(self.files) == {(d, t) for d in domains for t in TABLES}, "四表白名单缺失或重复")
        for (domain, table), row in self.files.items():
            require(row["repo_path"] == f"{domain}/formatted_tables/{table}.parquet", "不允许非标准表路径")
            require(re.fullmatch(r"[0-9a-f]{64}", row["expected_sha256"]), "缺少固定SHA256")
        self.verified = set()

    def table(self, domain, table):
        row = self.files[domain, table]
        path = self.root / row["repo_path"]
        require(path.resolve().is_relative_to(self.root), "公开表路径逃逸数据根目录")
        require(path.stat().st_size == row["expected_bytes"], f"{domain}/{table}: 表大小不匹配")
        require(file_sha256(path) == row["expected_sha256"], f"{domain}/{table}: SHA256不匹配")
        schema = pq.ParquetFile(path).schema_arrow
        require(schema.equals(SCHEMAS[table], check_metadata=False), f"{domain}/{table}: 官方v3 schema不匹配")
        require((schema.metadata or {}).get(b"measurement_db.schema_version") == b"3", "Parquet不是schema v3")
        self.verified.add((domain, table))
        return pl.scan_parquet(path)

    def load_group(self, entry):
        domain, subject_id = entry["domain"], entry["subject_id"]
        catalog = self.table(domain, "benchmarks").collect()
        require(catalog.height == 1, "每域必须一个benchmark")
        benchmark = catalog.row(0, named=True)
        require(all(benchmark[k] == v for k, v in
                    (("release", "public"), ("response_type", "binary"), ("granularity", "item"))),
                "catalog不是公共binary item")
        scale = json.loads(benchmark["response_scale"])
        require(scale.get("kind") == "discrete" and sorted(scale.get("values", [])) == [0, 1], "不是0/1标度")
        subjects = self.table(domain, "subjects").select("subject_id", *SUBJECT_FIELDS).collect()
        require(subjects["subject_id"].null_count() == 0 and subjects["subject_id"].n_unique() == subjects.height,
                "subject主键缺失或重复")
        selected = subjects.filter(pl.col("subject_id") == subject_id)
        require(selected.height == 1, "选定公共subject不存在")
        subject = VisibleSubject(**{name: text(selected[name][0]) for name in SUBJECT_FIELDS})
        responses = self.table(domain, "responses").filter(pl.col("subject_id") == subject_id).collect(engine="streaming")
        for name in ("subject_id", "item_id", "response_id", "benchmark_id", "trial"):
            require(responses[name].null_count() == 0, f"response必填字段{name}含NULL")
        require(responses["response_id"].n_unique() == responses.height, "response_id重复")
        require(responses.select(pl.struct("subject_id", "item_id", "trial", "test_condition", "interactors").n_unique()).item()
                == responses.height, "response观测键重复")
        require(responses.filter((pl.col("trial") < 1) |
                (pl.col("benchmark_id") != benchmark["benchmark_id"])).height == 0, "response外键或trial不符")
        require(responses.filter(pl.col("response").is_not_null() &
                (~pl.col("response").is_finite() | ~pl.col("response").is_in([0.0, 1.0]))).height == 0,
                "存在非有限或非二元成绩")
        items_scan = self.table(domain, "items")
        bank = items_scan.select("item_id", "benchmark_id").collect()
        require(bank["item_id"].null_count() == 0 and bank["item_id"].n_unique() == bank.height, "item主键缺失或重复")
        require(bank["benchmark_id"].null_count() == 0 and
                bank.filter(pl.col("benchmark_id") != benchmark["benchmark_id"]).height == 0, "item benchmark外键不符")
        used = responses.select("item_id").unique()
        require(used.join(bank.select("item_id"), on="item_id", how="anti").height == 0, "response item外键缺失")
        item_rows = items_scan.select("item_id", "content", "item_features").join(
            used.lazy(), on="item_id", how="semi").collect(engine="streaming")
        item_text = {row["item_id"]: (text(row["content"]), text(row["item_features"]).strip())
                     for row in item_rows.iter_rows(named=True)}
        by_item = defaultdict(list)
        for row in responses.sort("subject_id", "item_id", "trial", "response_id").iter_rows(named=True):
            by_item[row["item_id"]].append(SnapshotResponse(
                None if row["response"] is None else int(row["response"]), text(row["interactors"]).strip(),
                ResponseProvenance(row["response_id"], row["trial"], row["test_condition"])))
        items = tuple(SnapshotItem(item_id, *item_text[item_id], self.aliases[domain], tuple(values))
                      for item_id, values in sorted(by_item.items()))
        require(subjects.height == benchmark["n_subjects"] and bank.height == benchmark["n_items"],
                "catalog与subject/item数量不一致")
        require(sum(any(r.value is not None for r in item.responses) for item in items)
                == entry["distinct_graded_items"], "选定pair完整已评分item数不匹配")
        require(sum(r.value is not None for item in items for r in item.responses)
                == entry["graded_responses"], "选定pair完整已评分response数不匹配")
        return SnapshotGroup(domain, benchmark["benchmark_id"], subject_id, subject, items)
