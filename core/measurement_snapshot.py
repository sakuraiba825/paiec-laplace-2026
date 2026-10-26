"""固定公开快照的只读加载器；源身份和标签只供审计，不进入可见字典。"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

REVISION = "54074453da7b9911be0f48e4c560be547bc84d21"
EXPECTED_DOMAIN_COUNT = 160
POLICY = "graded_only_explicit_local_assumption"
TABLES = ("benchmarks", "subjects", "items", "responses")
SUBJECT_FIELDS = (
    "normalized_name", "provider", "release_date", "access_date", "harness",
    "reasoning_effort", "harness_version", "subject_features_extra",
)
DEFAULT_SNAPSHOT = Path(__file__).resolve().parents[1] / "data/snapshots/measurement-db_54074453"


def _schema(fields: Sequence[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema([pa.field(*field) for field in fields])


# 对应已存档官方 parquet_schemas.yaml v3；额外列同样拒绝。
SCHEMAS = {
    "subjects": _schema([
        ("subject_id", pa.string(), False), ("display_name", pa.string(), False),
        *((name, pa.string(), True) for name in SUBJECT_FIELDS),
    ]),
    "items": _schema([
        ("item_id", pa.string(), False), ("benchmark_id", pa.string(), False),
        ("raw_item_id", pa.string(), False), ("content", pa.string(), True),
        ("asset_manifest", pa.string(), True), ("grading_criterion", pa.string(), False),
        ("verifier", pa.string(), False), ("content_hash", pa.string(), True),
        ("item_features", pa.string(), True),
    ]),
    "responses": _schema([
        ("subject_id", pa.string(), False), ("item_id", pa.string(), False),
        ("response_id", pa.string(), False), ("benchmark_id", pa.string(), False),
        ("trial", pa.int64(), False), ("test_condition", pa.string(), True),
        ("interactors", pa.string(), True), ("response", pa.float64(), True),
    ]),
    "benchmarks": _schema([
        ("benchmark_id", pa.string(), False), ("name", pa.string(), False),
        ("version", pa.string(), True), ("license", pa.string(), False),
        ("source_url", pa.string(), False), ("description", pa.string(), False),
        ("one_line_description", pa.string(), True),
        ("modality", pa.list_(pa.string()), False), ("domain", pa.list_(pa.string()), False),
        ("multi_single_turn", pa.string(), False), ("response_type", pa.string(), False),
        ("response_scale", pa.string(), False), ("categorical", pa.bool_(), False),
        ("paper_url", pa.string(), True), ("release_date", pa.string(), True),
        ("granularity", pa.string(), False), ("release", pa.string(), True),
        ("n_response_values", pa.int64(), True), ("benchmark_features", pa.string(), True),
        ("n_subjects", pa.int64(), False), ("n_items", pa.int64(), False),
        ("n_responses", pa.int64(), False), ("max_trial", pa.int64(), False),
        ("coverage", pa.float64(), False), ("has_reference_answer", pa.bool_(), False),
    ]),
}


class SnapshotValidationError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotValidationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: str | None) -> str:
    return "" if value is None else value


@dataclass(frozen=True, slots=True)
class VisibleSubject:
    normalized_name: str
    provider: str
    release_date: str
    access_date: str
    harness: str
    reasoning_effort: str
    harness_version: str
    subject_features_extra: str

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in SUBJECT_FIELDS}


@dataclass(frozen=True, slots=True)
class ResponseProvenance:
    response_id: str
    trial: int
    test_condition: str | None


@dataclass(frozen=True, slots=True)
class SnapshotResponse:
    value: int | None
    interactors: str
    provenance: ResponseProvenance


@dataclass(frozen=True, slots=True)
class SnapshotItem:
    source_item_id: str
    item_content: str
    item_features: str
    benchmark_id: str
    responses: tuple[SnapshotResponse, ...]

    def visible(self, response: SnapshotResponse) -> dict[str, str]:
        return {
            "item_content": self.item_content, "item_features": self.item_features,
            "benchmark_id": self.benchmark_id, "interactors": response.interactors,
        }


@dataclass(frozen=True, slots=True)
class SnapshotGroup:
    domain: str
    source_benchmark_id: str
    source_subject_id: str
    subject: VisibleSubject
    items: tuple[SnapshotItem, ...]


@dataclass(frozen=True, slots=True)
class GradedPairSummary:
    subject_id: str
    distinct_graded_items: int
    graded_responses: int


@dataclass(frozen=True, slots=True)
class DomainMetadata:
    domain: str
    source_benchmark_id: str
    benchmark_id: str
    name: str
    version: str | None
    modality: tuple[str, ...]
    categories: tuple[str, ...]
    response_scale: str
    direction: str | None
    release: str
    response_type: str
    granularity: str
    source_subject_count: int
    source_item_bank_count: int
    recorded_response_count: int
    null_response_count: int
    graded_response_count: int
    observed_pair_count: int
    pairs: tuple[GradedPairSummary, ...]
    subjects_without_graded_pair_count: int
    graded_cells_with_multiple_interactors: int
    visible_subject_collision_count: int
    pair_count_source: str = "content_validation_prior_audit"
    policy: str = POLICY


@dataclass(frozen=True, slots=True)
class PairLoadReport:
    source_subject_id: str
    recorded_response_count: int
    null_response_count: int
    graded_response_count: int
    source_distinct_item_count: int
    graded_distinct_item_count: int
    null_only_item_ids: tuple[str, ...]
    item_bank_unobserved_count: int
    item_bank_without_graded_count: int
    multiple_interactor_cells: int
    graded_multiple_interactor_cells: int
    multiple_condition_cells: int
    graded_multiple_condition_cells: int
    below_80_graded_items: bool


@dataclass(frozen=True, slots=True)
class VerifiedTable:
    table: str
    repo_path: str
    bytes: int
    sha256: str
    rows: int


@dataclass(frozen=True, slots=True)
class LoadReport:
    revision: str
    domain: str
    selected_subject_ids: tuple[str, ...]
    policy: str
    recorded_response_count: int
    null_response_count: int
    graded_response_count: int
    source_distinct_item_count: int
    graded_distinct_item_count: int
    source_subject_item_cells: int
    graded_subject_item_cells: int
    item_bank_count: int
    pair_count: int
    pairs_without_graded_responses: tuple[str, ...]
    domain_visible_subject_collision_count: int
    selected_visible_subject_collision_count: int
    visible_subject_collision_groups: tuple[tuple[str, ...], ...]
    multiple_interactor_cells: int
    graded_multiple_interactor_cells: int
    multiple_condition_cells: int
    graded_multiple_condition_cells: int
    pairs: tuple[PairLoadReport, ...]
    verified_tables: tuple[VerifiedTable, ...]
    manifest_sha256: str
    content_validation_sha256: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LoadedDomain:
    groups: tuple[SnapshotGroup, ...]
    report: LoadReport


class MeasurementSnapshot:
    """只按白名单取文件；加载单位是完整源 benchmark–subject pair。"""

    def __init__(self, snapshot_dir: str | Path = DEFAULT_SNAPSHOT):
        self.root = Path(snapshot_dir).resolve()
        documents = {}
        self.document_hashes = {}
        for name in ("download_plan", "download_manifest", "content_validation"):
            path = self.root / f"{name}.json"
            raw = path.read_bytes()
            documents[name] = json.loads(raw)
            self.document_hashes[name] = hashlib.sha256(raw).hexdigest()
            _require(documents[name].get("revision") == REVISION, f"{name}: revision 不匹配")
        plan = documents["download_plan"]
        manifest = documents["download_manifest"]
        audit = documents["content_validation"]
        _require(manifest.get("approved_plan_sha256") == self.document_hashes["download_plan"], "manifest未绑定当前已批准计划SHA256")
        for name, doc in (("manifest", manifest), ("content_validation", audit)):
            _require(doc.get("status") == "complete" and not doc.get("failures"), f"{name} 未完整验证")
        _require(plan.get("filter") == {"release": "public", "response_type": "binary", "granularity": "item"}, "不是已授权公开binary item范围")
        _require(plan.get("metadata_inventory_complete") is True and not plan.get("metadata_errors"), "计划元数据不完整")
        self.domains = tuple(plan["eligible_domains"])
        _require(len(self.domains) == len(set(self.domains)) == EXPECTED_DOMAIN_COUNT, "公开域白名单数量错误")
        _require(all(re.fullmatch(r"[a-z][a-z0-9_]*", d) and not d.startswith("private") for d in self.domains), "非法或private域")
        _require(plan.get("eligible_domain_count") == manifest.get("eligible_domain_count") == EXPECTED_DOMAIN_COUNT, "计划与清单域数量不一致")
        expected_keys = {(domain, table) for domain in self.domains for table in TABLES}
        self._planned = self._index_files(plan["files"], expected_keys)
        self._files = self._index_files(manifest["files"], expected_keys)
        audited = self._index_files(audit["files"], expected_keys)
        self._audits = {entry["domain"]: entry for entry in audit["domains"]}
        _require(len(audit["domains"]) == len(self._audits) and set(self._audits) == set(self.domains), "内容审计域不完整")
        for key, entry in self._files.items():
            planned = self._planned[key]
            proof = audited[key]
            domain, table = key
            repo_path = f"{domain}/formatted_tables/{table}.parquet"
            _require(entry.get("revision") == REVISION, f"{repo_path}: revision 不匹配")
            _require(entry.get("repo_path") == planned.get("repo_path") == repo_path, "非标准白名单路径")
            _require(entry.get("status") in {"downloaded", "verified_existing"} and entry.get("hash_verified") is True, f"{repo_path}: 表未验真")
            _require(entry.get("canonical_column_schema_match") is True and proof.get("schema_and_nonnullable_valid") is True, f"{repo_path}: schema未验证")
            digest = planned.get("expected_sha256", "")
            _require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), "缺少固定SHA256")
            _require(digest == entry.get("sha256") == entry.get("expected_sha256") == proof.get("sha256"), f"{repo_path}: SHA证据不一致")
            _require(planned.get("expected_bytes") == entry.get("bytes") and isinstance(entry.get("bytes"), int), f"{repo_path}: 大小证据不一致")
            _require(entry.get("rows") == proof.get("rows"), f"{repo_path}: 行数证据不一致")
        self._verified: dict[tuple[str, str], tuple[tuple[int, ...], VerifiedTable]] = {}
        self._aliases = {domain: f"local_benchmark_{index:04d}" for index, domain in enumerate(sorted(self.domains), 1)}

    @staticmethod
    def _index_files(entries: list[dict], expected: set[tuple[str, str]]) -> dict:
        result = {(row["domain"], row["table"]): row for row in entries}
        _require(len(entries) == len(result) and set(result) == expected, "四张核心表白名单不完整或有重复/额外表")
        return result

    def _path(self, domain: str, table: str) -> Path:
        _require(domain in self._aliases and table in TABLES, "请求不在公开核心白名单中")
        return self.root / domain / "formatted_tables" / f"{table}.parquet"

    def _verify_table(self, domain: str, table: str) -> VerifiedTable:
        path = self._path(domain, table)
        _require(path.resolve().is_relative_to(self.root), "表路径逃逸快照目录")
        entry = self._files[domain, table]
        stat = path.stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._verified.get((domain, table))
        if cached is not None and cached[0] == signature:
            return cached[1]
        _require(stat.st_size == entry["bytes"], f"{domain}/{table}: 实际大小不匹配")
        _require(_sha256(path) == entry["sha256"], f"{domain}/{table}: 实际SHA256不匹配")
        after = path.stat()
        _require(signature == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "验真期间文件发生变化")
        parquet = pq.ParquetFile(path)
        schema = parquet.schema_arrow
        _require(schema.equals(SCHEMAS[table], check_metadata=False), f"{domain}/{table}: 非官方v3列/type/nullability")
        _require((schema.metadata or {}).get(b"measurement_db.schema_version") == b"3", "Parquet不是schema v3")
        _require(parquet.metadata.num_rows == entry["rows"], "实际行数不匹配")
        proof = VerifiedTable(table, entry["repo_path"], entry["bytes"], entry["sha256"], entry["rows"])
        self._verified[domain, table] = (signature, proof)
        return proof

    def _scan(self, domain: str, table: str, columns: Sequence[str]) -> pl.LazyFrame:
        self._verify_table(domain, table)
        return pl.scan_parquet(self._path(domain, table)).select(list(columns))

    def _catalog(self, domain: str) -> dict:
        row = self._scan(domain, "benchmarks", SCHEMAS["benchmarks"].names).collect()
        _require(row.height == 1, "每域必须只有一个benchmark")
        value = row.row(0, named=True)
        _require(value["release"] == "public" and value["response_type"] == "binary" and value["granularity"] == "item", "catalog不在已授权公开binary item范围")
        scale = json.loads(value["response_scale"])
        _require(scale.get("kind") == "discrete" and sorted(scale.get("values", [])) == [0, 1], "binary响应标度不符")
        return value

    def list_domains(self) -> tuple[DomainMetadata, ...]:
        """只复验小catalog；pair数来自绑定相同SHA的既有完整审计，不重新扫描71M行。"""
        result = []
        for domain in self.domains:
            catalog = self._catalog(domain)
            audit = self._audits[domain]
            readiness = audit["visible_input_readiness"]
            pairs = tuple(GradedPairSummary(**pair) for pair in audit["pairs"])
            _require(len(pairs) == audit["pairs_with_at_least_one_graded_response"], "已评分pair审计数量不一致")
            result.append(DomainMetadata(
                domain=domain, source_benchmark_id=catalog["benchmark_id"], benchmark_id=self._aliases[domain],
                name=catalog["name"], version=catalog["version"], modality=tuple(catalog["modality"]),
                categories=tuple(catalog["domain"]), response_scale=catalog["response_scale"],
                direction=json.loads(catalog["response_scale"]).get("direction"), release=catalog["release"],
                response_type=catalog["response_type"], granularity=catalog["granularity"],
                source_subject_count=catalog["n_subjects"], source_item_bank_count=catalog["n_items"],
                recorded_response_count=audit["recorded_response_rows"], null_response_count=audit["ungraded_response_rows"],
                graded_response_count=audit["binary_grade_rows"], observed_pair_count=audit["observed_subject_pair_count"],
                pairs=pairs, subjects_without_graded_pair_count=catalog["n_subjects"] - len(pairs),
                graded_cells_with_multiple_interactors=readiness["graded_cells_with_multiple_visible_interactors"],
                visible_subject_collision_count=readiness["source_subjects_minus_visible_dicts"],
            ))
        return tuple(result)

    def load_domain(self, domain: str, subject_ids: Sequence[str] | None = None) -> LoadedDomain:
        """不提供item/condition过滤或行数上限；所选subject的全部已记录响应都保留。"""
        catalog = self._catalog(domain)
        benchmark = catalog["benchmark_id"]
        subjects = self._scan(domain, "subjects", ("subject_id", *SUBJECT_FIELDS)).collect()
        _require(subjects["subject_id"].null_count() == 0 and subjects["subject_id"].n_unique() == subjects.height, "subject主键缺失或重复")
        visible_subjects = {
            row["subject_id"]: VisibleSubject(**{name: _text(row[name]) for name in SUBJECT_FIELDS})
            for row in subjects.iter_rows(named=True)
        }
        if subject_ids is None:
            selected = tuple(sorted(visible_subjects))
        else:
            _require(not isinstance(subject_ids, (str, bytes)), "subject_ids必须是ID序列")
            selected = tuple(sorted(subject_ids))
            _require(len(selected) == len(set(selected)) and len(selected) > 0, "subject_ids不能为空或重复")
            _require(set(selected).issubset(visible_subjects), "请求subject不存在")
        responses_scan = self._scan(domain, "responses", SCHEMAS["responses"].names)
        if subject_ids is not None:
            responses_scan = responses_scan.filter(pl.col("subject_id").is_in(selected))
        responses = responses_scan.collect(engine="streaming")
        for name in ("subject_id", "item_id", "response_id", "benchmark_id", "trial"):
            _require(responses[name].null_count() == 0, f"response必填字段{name}含NULL")
        _require(responses["response_id"].n_unique() == responses.height, "response_id重复")
        _require(responses.select(pl.struct("subject_id", "item_id", "trial", "test_condition", "interactors").n_unique()).item() == responses.height, "response观测键重复")
        _require(not responses.filter((pl.col("trial") < 1) | (pl.col("benchmark_id") != benchmark)).height, "response trial/benchmark外键不符")
        _require(set(responses["subject_id"].unique()).issubset(visible_subjects), "response subject外键缺失")
        invalid = responses.filter(pl.col("response").is_not_null() & (~pl.col("response").is_finite() | ~pl.col("response").is_in([0.0, 1.0])))
        _require(invalid.height == 0, "存在非有限或非精确0/1成绩")
        bank = self._scan(domain, "items", ("item_id", "benchmark_id")).collect()
        _require(bank["item_id"].null_count() == 0 and bank["item_id"].n_unique() == bank.height, "item主键缺失或重复")
        _require(bank["benchmark_id"].null_count() == 0 and not bank.filter(pl.col("benchmark_id") != benchmark).height, "item benchmark外键不符")
        used = responses.select("item_id").unique()
        _require(used.join(bank.select("item_id"), on="item_id", how="anti").height == 0, "response item外键缺失")
        # 大文本只为已选pair涉及的源item加载一次，不同subject复用同一个字符串对象。
        item_rows = self._scan(domain, "items", ("item_id", "content", "item_features")).join(used.lazy(), on="item_id", how="semi").collect(engine="streaming")
        item_text = {row["item_id"]: (_text(row["content"]), _text(row["item_features"]).strip()) for row in item_rows.iter_rows(named=True)}
        rows_by_subject: dict[str, dict[str, list[SnapshotResponse]]] = defaultdict(lambda: defaultdict(list))
        string_pool: dict[str, str] = {}
        for row in responses.sort("subject_id", "item_id", "trial", "response_id").iter_rows(named=True):
            interactor = _text(row["interactors"]).strip()
            interactor = string_pool.setdefault(interactor, interactor)
            condition = row["test_condition"]
            if condition is not None:
                condition = string_pool.setdefault(condition, condition)
            rows_by_subject[row["subject_id"]][row["item_id"]].append(SnapshotResponse(
                None if row["response"] is None else int(row["response"]), interactor,
                ResponseProvenance(row["response_id"], row["trial"], condition),
            ))
        groups = []
        pair_reports = []
        graded_item_ids = set()
        for subject_id in selected:
            items = tuple(SnapshotItem(item_id, *item_text[item_id], self._aliases[domain], tuple(values)) for item_id, values in sorted(rows_by_subject[subject_id].items()))
            groups.append(SnapshotGroup(domain, benchmark, subject_id, visible_subjects[subject_id], items))
            recorded = nulls = multi_interactors = graded_multi_interactors = multi_conditions = graded_multi_conditions = 0
            null_only = []
            graded_distinct = 0
            for item in items:
                graded = tuple(r for r in item.responses if r.value is not None)
                recorded += len(item.responses)
                nulls += len(item.responses) - len(graded)
                if graded:
                    graded_distinct += 1
                    graded_item_ids.add(item.source_item_id)
                else:
                    null_only.append(item.source_item_id)
                multi_interactors += len({r.interactors for r in item.responses}) > 1
                graded_multi_interactors += len({r.interactors for r in graded}) > 1
                multi_conditions += len({r.provenance.test_condition for r in item.responses}) > 1
                graded_multi_conditions += len({r.provenance.test_condition for r in graded}) > 1
            pair_reports.append(PairLoadReport(
                subject_id, recorded, nulls, recorded - nulls, len(items), graded_distinct,
                tuple(null_only), bank.height - len(items), bank.height - graded_distinct,
                multi_interactors, graded_multi_interactors, multi_conditions, graded_multi_conditions,
                graded_distinct < 80,
            ))
        _require(subjects.height == catalog["n_subjects"] and bank.height == catalog["n_items"], "catalog与实体表数量不一致")
        audit = self._audits[domain]
        audited_pairs = {pair["subject_id"]: pair for pair in audit["pairs"]}
        for pair in pair_reports:
            expected = audited_pairs.get(pair.source_subject_id, {"distinct_graded_items": 0, "graded_responses": 0})
            _require(pair.graded_distinct_item_count == expected["distinct_graded_items"] and pair.graded_response_count == expected["graded_responses"], "完整pair与已验审计数量不一致")
        if subject_ids is None:
            _require(responses.height == catalog["n_responses"] == audit["recorded_response_rows"], "完整域response数量不一致")
            _require(responses["response"].null_count() == audit["ungraded_response_rows"], "完整域NULL数量不一致")
        collisions: dict[VisibleSubject, list[str]] = defaultdict(list)
        for subject_id, visible in visible_subjects.items():
            collisions[visible].append(subject_id)
        collision_groups = tuple(sorted(tuple(sorted(ids)) for ids in collisions.values() if len(ids) > 1))
        selected_counts = Counter(visible_subjects[sid] for sid in selected)
        proof = tuple(self._verify_table(domain, table) for table in TABLES)
        null_count = responses["response"].null_count()
        report = LoadReport(
            REVISION, domain, selected, POLICY, responses.height, null_count, responses.height - null_count,
            used.height, len(graded_item_ids), sum(p.source_distinct_item_count for p in pair_reports),
            sum(p.graded_distinct_item_count for p in pair_reports), bank.height, len(groups),
            tuple(p.source_subject_id for p in pair_reports if p.graded_response_count == 0),
            len(visible_subjects) - len(collisions), len(selected) - len(selected_counts), collision_groups,
            sum(p.multiple_interactor_cells for p in pair_reports), sum(p.graded_multiple_interactor_cells for p in pair_reports),
            sum(p.multiple_condition_cells for p in pair_reports), sum(p.graded_multiple_condition_cells for p in pair_reports),
            tuple(pair_reports), proof, self.document_hashes["download_manifest"], self.document_hashes["content_validation"],
        )
        return LoadedDomain(tuple(groups), report)
