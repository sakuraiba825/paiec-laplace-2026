"""逐item的最小本地ALC；仅规范合成输入，不是官方coordinator复刻。

只执行已审阅的可信model.py，本进程runner不是安全沙箱。默认采集公式及
公开契约来自固定starting-kit；split、顺序、response选择和调度均为本地假设。
本模块不读取真实数据，不恢复聚合trials，不写文件，也不覆盖v2研究入口。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Mapping
import zipfile

PROTOCOL_VERSION = "alc-item-stream-v3-local-assumptions-v1"
BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness", "reasoning_effort", "harness_version", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")
BELOW80_POLICIES = ("exclude_with_report", "diagnostic_no_acquisition")
SELECTION_MODES = ("full_public_transfer_evaluation", "formative_style_whole_group_draw_diagnostic")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _identity(value):
    if type(value) is not str or not value:
        raise ValueError("内部身份必须是非空稳定字符串，不能猜测或补齐")


def _fields(value, fields):
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise ValueError("可见字典字段必须与官方接口完全一致")
    if any(type(v) is not str for v in value.values()):
        raise ValueError("每个可见字段必须是原生字符串")
    return MappingProxyType({key: value[key] for key in fields})


@dataclass(frozen=True)
class ResponseRecord:
    key: str
    outcome: int

    def __post_init__(self):
        _identity(self.key)
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise ValueError("仅接受原始二元int记录；NULL、均值、布尔或字符串须由经审阅loader处理")


@dataclass(frozen=True)
class Item:
    key: str
    visible: Mapping[str, str]
    responses: tuple[ResponseRecord, ...]

    def __post_init__(self):
        _identity(self.key)
        object.__setattr__(self, "visible", _fields(self.visible, ITEM_FIELDS))
        records = tuple(self.responses)
        if not records or any(not isinstance(r, ResponseRecord) for r in records):
            raise ValueError("每item须有明确的原始ResponseRecord，不能从均值生成")
        if len({r.key for r in records}) != len(records):
            raise ValueError("同item的response key重复")
        object.__setattr__(self, "responses", tuple(sorted(records, key=lambda r: r.key)))


@dataclass(frozen=True)
class Pair:
    key: str
    benchmark_key: str
    subject_key: str
    subject: Mapping[str, str]
    items: tuple[Item, ...]

    def __post_init__(self):
        for key in (self.key, self.benchmark_key, self.subject_key):
            _identity(key)
        object.__setattr__(self, "subject", _fields(self.subject, SUBJECT_FIELDS))
        items = tuple(self.items)
        if not items or any(not isinstance(item, Item) for item in items):
            raise ValueError("pair须有规范Item；空group不能评分")
        if len({item.key for item in items}) != len(items):
            raise ValueError("pair内source item身份重复")
        if len({item.visible["benchmark_id"] for item in items}) != 1 or not items[0].visible["benchmark_id"]:
            raise ValueError("同pair的benchmark alias必须非空且一致")
        object.__setattr__(self, "items", tuple(sorted(items, key=lambda item: item.key)))


@dataclass(frozen=True)
class Assumptions:
    split_seed: int
    order_seed: int
    response_seed: int
    response_choice: str

    def __post_init__(self):
        if any(type(seed) is not int for seed in (self.split_seed, self.order_seed, self.response_seed)):
            raise ValueError("本地seed必须显式整数")
        if self.response_choice not in ("hash_rank_record", "first_record_key"):
            raise ValueError("response choice必须是已命名本地假设，不能声称官方规则")

    def report(self):
        return {
            **asdict(self),
            "split_rule": "sha256_rank_pair_item_key_ceil_half_acquisition",
            "order_rule": "sha256_rank_pair_item_key_independent_of_responses",
            "scheduler": "sorted_pair_key_with_global_checkpoint_barrier",
            "subject_context_alias": "local_subject_加subject_key的sha256前16位；非线上匿名ID",
            "response_rule": "hash_rank_record按opaque record key的固定哈希rank取首条；first_record_key按key字典序取首条；都不依赖outcome",
            "draw_count_unit": "sum(pair内distinct source items)；本地subject-item计数假设，非已知服务端精确去重细节",
            "coordinator_details_verified": False,
            "seeds_are_independent_domains": False,
        }


@dataclass(frozen=True)
class SelectionManifest:
    mode: str
    pair_keys: tuple[str, ...]
    provenance: str

    def __post_init__(self):
        if self.mode not in SELECTION_MODES:
            raise ValueError("必须明确全公开迁移层或formative-style诊断层")
        _identity(self.provenance)
        keys = tuple(self.pair_keys)
        for key in keys:
            _identity(key)
        if len(set(keys)) != len(keys):
            raise ValueError("selection manifest不能重复group")
        object.__setattr__(self, "pair_keys", tuple(sorted(keys)))


def select_pairs(pairs, selection):
    pairs = tuple(pairs)
    by_key = {p.key: p for p in pairs}
    if len(by_key) != len(pairs):
        raise ValueError("pair key重复")
    if len({(p.benchmark_key, p.subject_key) for p in pairs}) != len(pairs):
        raise ValueError("相同benchmark-subject group不能拆成多个pair")
    if not set(selection.pair_keys) <= set(by_key):
        raise ValueError("selection引用不存在的pair")
    if selection.mode == SELECTION_MODES[0] and set(selection.pair_keys) != set(by_key):
        raise ValueError("full-public模式不能悄悄删除group")
    selected = tuple(by_key[key] for key in selection.pair_keys)
    if selection.mode == SELECTION_MODES[1] and sum(len(p.items) for p in selected) > 1000:
        raise ValueError("完整group诊断超过本地1000 source subject-item cap，不能截组")
    return selected


def _snapshot(pair):
    return _digest([pair.key, pair.benchmark_key, pair.subject_key, dict(pair.subject),
                    [[item.key, dict(item.visible), [[r.key, r.outcome] for r in item.responses]] for item in pair.items]])


@dataclass(frozen=True)
class SplitManifest:
    pair_key: str
    snapshot_sha256: str
    split_seed: int
    below80_policy: str
    acquisition_keys: tuple[str, ...]
    evaluation_keys: tuple[str, ...]
    fingerprint: str
    protocol_id: str = PROTOCOL_VERSION

    def __post_init__(self):
        object.__setattr__(self, "acquisition_keys", tuple(self.acquisition_keys))
        object.__setattr__(self, "evaluation_keys", tuple(self.evaluation_keys))


def split_pair(pair, assumptions, *, below80_policy, previous=None):
    if below80_policy not in BELOW80_POLICIES:
        raise ValueError("below80须显式本地处置，不得冒充官方排除")
    snapshot = _snapshot(pair)
    if previous is not None and previous.snapshot_sha256 != snapshot:
        raise NotImplementedError("不支持snapshot新增、删除或修改；须独立新snapshot实验，不能移动旧assignment")
    if len(pair.items) >= 80:
        ranked = sorted(pair.items, key=lambda item: (_digest([PROTOCOL_VERSION, "split", assumptions.split_seed, pair.key, item.key]), item.key))
        cut = (len(ranked) + 1) // 2
        acquisition = tuple(sorted(item.key for item in ranked[:cut]))
        evaluation = tuple(sorted(item.key for item in ranked[cut:]))
    else:
        acquisition = ()
        evaluation = tuple(item.key for item in pair.items) if below80_policy == "diagnostic_no_acquisition" else ()
    payload = [PROTOCOL_VERSION, pair.key, snapshot, assumptions.split_seed, below80_policy, acquisition, evaluation]
    result = SplitManifest(pair.key, snapshot, assumptions.split_seed, below80_policy, acquisition, evaluation, _digest(payload))
    if previous is not None and previous != result:
        raise ValueError("previous split被篡改或配置已改变；不能静默重写")
    return result


def order_items(items, pair_key, assumptions):
    return tuple(sorted(items, key=lambda item: (_digest([PROTOCOL_VERSION, "order", assumptions.order_seed, pair_key, item.key]), item.key)))


def choose_response(item, pair_key, assumptions):
    if assumptions.response_choice == "first_record_key":
        return item.responses[0]
    return min(item.responses, key=lambda r: (_digest([PROTOCOL_VERSION, "response", assumptions.response_seed, pair_key, item.key, r.key]), r.key))


def default_query(input, context):
    # 逐字保留starter streaming_ingestion.py:177-181的公式及ensure_ascii默认值。
    key = json.dumps([context, input], sort_keys=True, separators=(",", ":"))
    uniform = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2 ** 64
    return uniform < min(1, context["labels_remaining"] / context["items_remaining"])


def _probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("预测须为有限[0,1]概率，不做静默兜底")
    return value


def response_brier(items, probabilities_by_item_key):
    items = tuple(items)
    if not items:
        raise ValueError("没有evaluation responses，不能生成分数")
    probabilities = {item.key: _probability(probabilities_by_item_key[item.key]) for item in items}
    return math.fsum((probabilities[item.key] - r.outcome) ** 2 for item in items for r in item.responses) / sum(len(item.responses) for item in items)


def aggregate_pair_scores(scores_by_pair, benchmark_by_pair):
    if not scores_by_pair:
        return {"primary_pair_mean": None, "benchmark_macro_diagnostic": None}
    groups = {}
    for key in sorted(scores_by_pair):
        groups.setdefault(benchmark_by_pair[key], []).append(scores_by_pair[key])
    return {
        "primary_pair_mean": math.fsum(scores_by_pair[key] for key in sorted(scores_by_pair)) / len(scores_by_pair),
        "benchmark_macro_diagnostic": math.fsum(math.fsum(values) / len(values) for _, values in sorted(groups.items())) / len(groups),
    }


@dataclass(frozen=True)
class ModelSource:
    """只读载入已人工审阅的可信代码；不提供文件/网络或进程隔离。"""
    source: str
    name: str
    archive_sha256: str | None = None

    @property
    def source_sha256(self):
        return hashlib.sha256(self.source.encode()).hexdigest()

    @property
    def candidate_sha256(self):
        return self.archive_sha256 or self.source_sha256

    @classmethod
    def from_flat_zip(cls, path):
        content = Path(path).read_bytes()
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            if archive.namelist() != ["model.py"]:
                raise NotImplementedError("最小可信runner仅支持单model.py包；不是官方ZIP格式限制")
            source = archive.read("model.py").decode("utf-8")
        return cls(source, Path(path).name, hashlib.sha256(content).hexdigest())

    def load_predict(self):
        module = ModuleType("local_v3_candidate")
        exec(compile(self.source, self.name, "exec"), module.__dict__)
        predict = module.__dict__.get("predict")
        if not callable(predict):
            raise ValueError("model.py须定义predict(input,labeled)")
        return predict


def _visible_input(pair, item):
    return [dict(pair.subject), dict(item.visible)]


def _validate_aliases(pairs):
    benchmarks, aliases, subjects = {}, {}, {}
    for pair in pairs:
        alias = pair.items[0].visible["benchmark_id"]
        if benchmarks.setdefault(pair.benchmark_key, alias) != alias or aliases.setdefault(alias, pair.benchmark_key) != pair.benchmark_key:
            raise ValueError("benchmark与本地alias须一一对应且稳定")
        if subjects.setdefault(pair.subject_key, dict(pair.subject)) != dict(pair.subject):
            raise ValueError("同subject key的八字段冲突；不同harness应为不同subject")


def evaluate(pairs, model_source, *, assumptions, selection, below80_policy, previous_splits=None):
    """计算可信合成实验；返回内部审计trace，trace不能作为participant输入。"""
    pairs = tuple(pairs)
    _validate_aliases(pairs)
    selected = select_pairs(pairs, selection)
    if below80_policy not in BELOW80_POLICIES:
        raise ValueError("below80须明确本地策略")
    if previous_splits is not None and set(previous_splits) != {p.key for p in selected}:
        raise ValueError("previous_splits必须完整匹配已选pair，不能部分缺失后静默新建")
    previous_splits = {} if previous_splits is None else dict(previous_splits)
    splits = {p.key: split_pair(p, assumptions, below80_policy=below80_policy, previous=previous_splits.get(p.key)) for p in selected}
    active = tuple(p for p in selected if len(p.items) >= 80)
    small = tuple(p for p in selected if len(p.items) < 80)
    scored = tuple(p for p in selected if splits[p.key].evaluation_keys)
    acquisition_keys = {p.key: set(splits[p.key].acquisition_keys) for p in active}
    evaluation_keys = {p.key: set(splits[p.key].evaluation_keys) for p in scored}
    streams = {p.key: order_items(tuple(i for i in p.items if i.key in acquisition_keys[p.key]), p.key, assumptions) for p in active}
    eval_items = {p.key: tuple(i for i in p.items if i.key in evaluation_keys[p.key]) for p in scored}
    cursors = {p.key: 0 for p in active}
    acquired = {p.key: 0 for p in selected}
    evidence, checkpoints, events = [], [], []
    imports = 0
    for cap in BUDGETS:
        for pair in active:
            stream = streams[pair.key]
            while acquired[pair.key] < cap and cursors[pair.key] < len(stream):
                cursor = cursors[pair.key]
                item = stream[cursor]
                context = {
                    "subject_id": "local_subject_" + _digest(pair.subject_key)[:16],
                    "benchmark_id": item.visible["benchmark_id"],
                    "labels_remaining": 31 - acquired[pair.key],
                    "labels_acquired": acquired[pair.key],
                    "max_labels": 31,
                    "items_remaining": len(stream) - cursor,
                }
                visible = _visible_input(pair, item)
                query = default_query(visible, context)
                event = {"event": "acquire", "pair_key": pair.key, "item_key": item.key, "cap": cap, "query": query, "context": context}
                cursors[pair.key] += 1
                if query:
                    response = choose_response(item, pair.key, assumptions)
                    evidence.append([visible, response.outcome])
                    acquired[pair.key] += 1
                    event.update(response_key=response.key, outcome=response.outcome)
                events.append(event)
        # 在全部pair达到当前cap后冻结；下一cap只能在所有预测计分承诺后继续。
        frozen = copy.deepcopy(evidence)
        predictions, cache, briers = [], {}, {}
        if scored:
            predict = model_source.load_predict()
            imports += 1
            for pair in scored:
                probabilities = {}
                for item in eval_items[pair.key]:
                    visible = _visible_input(pair, item)
                    key = _json(visible)
                    if key not in cache:
                        cache[key] = _probability(predict(copy.deepcopy(visible), copy.deepcopy(frozen)))
                    probabilities[item.key] = cache[key]
                    predictions.append({"pair_key": pair.key, "item_key": item.key, "probability": cache[key]})
                briers[pair.key] = response_brier(eval_items[pair.key], probabilities)
        checkpoint = {
            "cap": cap, "evidence": frozen, "evidence_sha256": _digest(frozen),
            "actual_labels_by_pair": dict(acquired), "predictions": predictions,
            "prediction_call_count": len(cache),
            "scored_response_count": sum(len(i.responses) for items in eval_items.values() for i in items),
            "pair_briers": briers,
        }
        checkpoints.append(checkpoint)
        events.append({"event": "checkpoint_committed", "cap": cap, "evidence_sha256": checkpoint["evidence_sha256"], "predictions_sha256": _digest(predictions)})
    pair_alcs = {p.key: math.fsum(weight * c["pair_briers"][p.key] for weight, c in zip(ALC_WEIGHTS, checkpoints)) for p in scored}
    primary_scores = {p.key: pair_alcs[p.key] for p in active}
    assumption_report = assumptions.report()
    return {
        "protocol_id": PROTOCOL_VERSION,
        "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_anchor": {"starting_kit_zip_sha256": "a870970a7b9e82385929db37435aac29e3f0ee648cce7df143787b88ac4770f6", "streaming_sha256": "682602ea065c5fec1731c223faae3514f42d764d96115b56ea3c8019832bab19", "baseline_commit": "82d330ddcdb16016a3ae9e048db7e588ba2c6a39"},
        "scope": "规范合成输入和可信模型的本地契约验证；非官方服务端复刻或模型性能证据",
        "schema_revision": "synthetic_explicit_records_v1",
        "dataset_revision": None,
        "input_snapshot_sha256": _digest([[p.key, _snapshot(p)] for p in sorted(pairs, key=lambda p: p.key)]),
        "assumptions": assumption_report,
        "assumptions_sha256": _digest(assumption_report),
        "selection_mode": selection.mode, "selection": asdict(selection),
        "input_pair_count": len(pairs), "selected_pair_count": len(selected),
        "selected_pair_keys": [p.key for p in selected],
        "selected_source_subject_item_count": sum(len(p.items) for p in selected),
        "selected_response_count": sum(len(i.responses) for p in selected for i in p.items),
        "primary_pair_count": len(active),
        "not_selected_pair_keys": sorted(set(p.key for p in pairs) - set(selection.pair_keys)),
        "excluded_pair_keys": [p.key for p in small],
        "below80_policy": below80_policy,
        "below80_diagnostic": {
            "official_behavior": "unknown",
            "note": "小组不进入主指标；diagnostic策略无自身采集、全item评分，仍可见其他已选pair的合法证据；不是官方行为断言",
            "pair_metrics": [{"pair_key": p.key, "brier_alc": pair_alcs[p.key]} for p in small if p.key in pair_alcs],
        },
        "split_manifests": [asdict(splits[p.key]) for p in selected],
        "candidate_sha256": model_source.candidate_sha256, "model_source_sha256": model_source.source_sha256,
        "model_import_count": imports, "budgets": list(BUDGETS), "alc_weights": list(ALC_WEIGHTS),
        "checkpoints": checkpoints, "events": events, "pair_alcs": primary_scores,
        **aggregate_pair_scores(primary_scores, {p.key: p.benchmark_key for p in active}),
        "ece": "not_computed",
        "capabilities": {
            "persistent_same_snapshot_reuse": True, "persistent_snapshot_extension": False,
            "real_data_loader": False, "custom_acquisition": False, "automatic_group_sampler": False,
            "parallel_workers": False, "candidate_container_isolation": False,
            "per_call_timeout": False, "per_response_visible_inputs": False,
            "multi_file_submission": False, "ece": False,
        },
    }
