"""公开已评分原始response的独立本地契约；不复刻官方coordinator。

按source item分池、排序和最多一次采集；先盲选response，再使用该条的
interactors查询。评估按完整可见input复用预测，所有response等权。
只执行已审阅可信model.py；本模块不是沙箱，不读取数据、不处理NULL。
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

from .alc_harness_v3 import (
    ALC_WEIGHTS, BELOW80_POLICIES, BUDGETS, ITEM_FIELDS, SELECTION_MODES,
    SUBJECT_FIELDS, Assumptions, ModelSource, SelectionManifest,
    _digest, _fields, _identity, _json, _probability,
    aggregate_pair_scores, default_query, select_pairs,
)

PROTOCOL_VERSION = "alc-response-record-stream-v3-local-assumptions-v1"
TRACE_LEVELS = ("full", "summary")


@dataclass(frozen=True)
class Record:
    key: str
    outcome: int
    interactors: str

    def __post_init__(self):
        _identity(self.key)
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise ValueError("仅接受已评分原始二元int；NULL、均值、布尔、字符串不能进入评分契约")
        if type(self.interactors) is not str:
            raise ValueError("response interactors须为loader规范化后的原生字符串")


@dataclass(frozen=True)
class Item:
    key: str
    base_visible: Mapping[str, str]
    responses: tuple[Record, ...]

    def __post_init__(self):
        _identity(self.key)
        object.__setattr__(self, "base_visible", _fields(self.base_visible, ITEM_FIELDS))
        if self.base_visible["interactors"] != "":
            raise ValueError("base_visible.interactors必须为空；实际值仅来自逐response记录")
        records = tuple(self.responses)
        if not records or any(not isinstance(r, Record) for r in records):
            raise ValueError("每item须有明确的原始Record；不能从均值生成或保留空评分item")
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
        if len({item.base_visible["benchmark_id"] for item in items}) != 1 or not items[0].base_visible["benchmark_id"]:
            raise ValueError("同pair的benchmark alias必须非空且一致")
        object.__setattr__(self, "items", tuple(sorted(items, key=lambda item: item.key)))


def visible_input(pair, item, record):
    """仅返回八个subject及四个item字符串字段；不暴露内部身份或标签。"""
    return [dict(pair.subject), {**item.base_visible, "interactors": record.interactors}]


def _snapshot(pair):
    return _digest([pair.key, pair.benchmark_key, pair.subject_key, dict(pair.subject),
                    [[i.key, dict(i.base_visible), [[r.key, r.outcome, r.interactors] for r in i.responses]] for i in pair.items]])


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
    if previous is not None and previous.protocol_id != PROTOCOL_VERSION:
        raise ValueError("previous split协议版本不匹配；不能复用旧v3状态")
    snapshot = _snapshot(pair)
    if previous is not None and previous.snapshot_sha256 != snapshot:
        raise NotImplementedError("不支持snapshot新增、删除或修改；须独立新snapshot实验")
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


def response_brier(items, probabilities_by_record_key):
    """概率字典的key为(item.key, record.key)，每条record独立等权。"""
    items = tuple(items)
    count = sum(len(i.responses) for i in items)
    if not count:
        raise ValueError("没有evaluation responses，不能生成分数")
    return math.fsum((_probability(probabilities_by_record_key[(item.key, r.key)]) - r.outcome) ** 2
                     for item in items for r in item.responses) / count


def _validate_aliases(pairs):
    benchmarks, aliases, subjects = {}, {}, {}
    for pair in pairs:
        alias = pair.items[0].base_visible["benchmark_id"]
        if benchmarks.setdefault(pair.benchmark_key, alias) != alias or aliases.setdefault(alias, pair.benchmark_key) != pair.benchmark_key:
            raise ValueError("benchmark与本地alias须一一对应且稳定")
        if subjects.setdefault(pair.subject_key, dict(pair.subject)) != dict(pair.subject):
            raise ValueError("同subject key的八字段冲突；不同harness应为不同subject")


def evaluate(pairs, model_source, *, assumptions, selection, below80_policy, previous_splits=None, trace_level="full"):
    """评估已审阅公开记录；summary仅减审计输出，不缩减每call的合法labeled。"""
    if trace_level not in TRACE_LEVELS:
        raise ValueError("trace_level必须是full或summary")
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
    response_counts = {p.key: sum(len(i.responses) for i in eval_items[p.key]) for p in scored}
    evaluation_records = {
        p.key: {
            "source_item_count": len(eval_items[p.key]), "scored_response_count": response_counts[p.key],
            "ordered_record_input_sha256": _digest([[i.key, r.key, _digest(visible_input(p, i, r))]
                                                     for i in eval_items[p.key] for r in i.responses]),
            "ordered_record_label_sha256": _digest([[i.key, r.key, r.outcome]
                                                     for i in eval_items[p.key] for r in i.responses]),
        } for p in scored
    }
    cursors = {p.key: 0 for p in active}
    acquired = {p.key: 0 for p in selected}
    successes = {p.key: 0 for p in selected}
    acquired_records = {p.key: [] for p in selected}
    global_acquired_records = []
    evidence, checkpoints, events = [], [], []
    imports = 0
    for cap in BUDGETS:
        for pair in active:
            stream = streams[pair.key]
            while acquired[pair.key] < cap and cursors[pair.key] < len(stream):
                cursor = cursors[pair.key]
                item = stream[cursor]
                response = choose_response(item, pair.key, assumptions)
                visible = visible_input(pair, item, response)
                context = {
                    "subject_id": "local_subject_" + _digest(pair.subject_key)[:16],
                    "benchmark_id": visible[1]["benchmark_id"],
                    "labels_remaining": 31 - acquired[pair.key],
                    "labels_acquired": acquired[pair.key],
                    "max_labels": 31,
                    "items_remaining": len(stream) - cursor,
                }
                query = default_query(visible, context)
                cursors[pair.key] += 1
                if query:
                    evidence.append([visible, response.outcome])
                    acquired[pair.key] += 1
                    successes[pair.key] += response.outcome
                    acquired_records[pair.key].append([item.key, response.key, response.outcome])
                    global_acquired_records.append([pair.key, item.key, response.key, response.outcome])
                if trace_level == "full":
                    event = {"event": "acquire", "pair_key": pair.key, "item_key": item.key,
                             "response_key": response.key, "input": visible, "cap": cap, "query": query, "context": context}
                    if query:
                        event["outcome"] = response.outcome
                    events.append(event)
        # 屏障之后才预测；每个call独立深拷贝，且下个cap前完成全部评分。
        frozen = copy.deepcopy(evidence)
        predictions, cache, briers = [], {}, {}
        if scored:
            predict = model_source.load_predict()
            imports += 1
            for pair in scored:
                def losses():
                    for item in eval_items[pair.key]:
                        for record in item.responses:
                            visible = visible_input(pair, item, record)
                            key = _json(visible)
                            if key not in cache:
                                cache[key] = _probability(predict(copy.deepcopy(visible), copy.deepcopy(frozen)))
                            probability = cache[key]
                            if trace_level == "full":
                                predictions.append({"pair_key": pair.key, "item_key": item.key,
                                                    "response_key": record.key, "probability": probability})
                            yield (probability - record.outcome) ** 2
                briers[pair.key] = math.fsum(losses()) / response_counts[pair.key]
        checkpoint = {
            "cap": cap, "evidence_sha256": _digest(frozen), "evidence_count": len(frozen),
            "actual_labels_by_pair": dict(acquired), "successes_by_pair": dict(successes),
            "acquired_record_sha256_by_pair": {key: _digest(records) for key, records in acquired_records.items()},
            "acquired_record_sha256": _digest(global_acquired_records),
            "candidate_items_seen_by_pair": dict(cursors),
            "prediction_call_count": len(cache), "scored_response_count": sum(response_counts.values()),
            "scored_response_count_by_pair": dict(response_counts), "pair_briers": briers,
        }
        if trace_level == "full":
            checkpoint.update(evidence=frozen, predictions=predictions)
        checkpoints.append(checkpoint)
        event = {"event": "checkpoint_committed", "cap": cap, "evidence_sha256": checkpoint["evidence_sha256"]}
        if trace_level == "full":
            event["predictions_sha256"] = _digest(predictions)
        events.append(event)
    pair_alcs = {p.key: math.fsum(weight * c["pair_briers"][p.key] for weight, c in zip(ALC_WEIGHTS, checkpoints)) for p in scored}
    primary_scores = {p.key: pair_alcs[p.key] for p in active}
    assumption_report = {
        **assumptions.report(),
        "hash_domain_protocol_id": PROTOCOL_VERSION,
        "response_input_rule": "先按不依赖outcome的record key规则选response，再将其interactors覆盖空base字段后执行默认query",
        "graded_population": "仅已评分0/1原始公开response；NULL原始行及NULL-only item由loader另行保留报告；非官方NULL处置断言",
        "evaluation_weight": "每条response等权；同checkpoint仅完整十二字段input相等时复用预测",
    }
    return {
        "protocol_id": PROTOCOL_VERSION, "trace_level": trace_level,
        "protocol_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_v3_source_sha256": hashlib.sha256(Path(__file__).with_name("alc_harness_v3.py").read_bytes()).hexdigest(),
        "official_anchor": {"starting_kit_zip_sha256": "a870970a7b9e82385929db37435aac29e3f0ee648cce7df143787b88ac4770f6", "streaming_sha256": "682602ea065c5fec1731c223faae3514f42d764d96115b56ea3c8019832bab19", "baseline_commit": "82d330ddcdb16016a3ae9e048db7e588ba2c6a39"},
        "scope": "公开已评分逐response输入和可信模型的本地契约；非官方服务端精确复刻，非线上成绩复现",
        "schema_revision": "graded_public_per_response_interactors_v1", "dataset_revision": None,
        "input_snapshot_sha256": _digest([[p.key, _snapshot(p)] for p in sorted(pairs, key=lambda p: p.key)]),
        "assumptions": assumption_report, "assumptions_sha256": _digest(assumption_report),
        "selection_mode": selection.mode, "selection": asdict(selection),
        "input_pair_count": len(pairs), "selected_pair_count": len(selected),
        "selected_pair_keys": [p.key for p in selected],
        "selected_source_subject_item_count": sum(len(p.items) for p in selected),
        "selected_response_count": sum(len(i.responses) for p in selected for i in p.items),
        "primary_pair_count": len(active),
        "not_selected_pair_keys": sorted(set(p.key for p in pairs) - set(selection.pair_keys)),
        "excluded_pair_keys": [p.key for p in small], "below80_policy": below80_policy,
        "below80_diagnostic": {
            "official_behavior": "unknown",
            "note": "小组不进入主指标；diagnostic无自身采集、全item评分，但可见其他已选pair合法证据；非官方行为断言",
            "pair_metrics": [{"pair_key": p.key, "brier_alc": pair_alcs[p.key]} for p in small if p.key in pair_alcs],
        },
        "split_manifests": [asdict(splits[p.key]) for p in selected],
        "evaluation_records_by_pair": evaluation_records,
        "record_fingerprint_schema": {
            "json": "json.dumps(sort_keys=True,separators=(',',':'),allow_nan=False)，默认ensure_ascii=True，UTF-8 SHA256",
            "evaluation_order": "按item.key及record.key字典序；input签名列表为[item_key,record_key,SHA256(full_visible_input)]，label签名列表为[item_key,record_key,outcome]",
            "acquisition_order": "实际购入顺序；pair签名列表为[item_key,record_key,outcome]，global签名列表额外以前缀pair_key标识；只含已公开采集标签",
            "replay": "在同snapshot/split/assumptions上，按cap和pair.key遍历；order_items→choose_response→visible_input→default_query；以31为总额，所有pair达到cap后冻结",
        },
        "split_counts_by_pair": {p.key: {"source_items": len(p.items), "acquisition_items": len(splits[p.key].acquisition_keys),
                                         "evaluation_items": len(splits[p.key].evaluation_keys), "evaluation_responses": response_counts.get(p.key, 0),
                                         "fingerprint": splits[p.key].fingerprint} for p in selected},
        "candidate_sha256": model_source.candidate_sha256, "model_source_sha256": model_source.source_sha256,
        "model_import_count": imports, "budgets": list(BUDGETS), "alc_weights": list(ALC_WEIGHTS),
        "checkpoints": checkpoints, "events": events, "pair_alcs": primary_scores,
        **aggregate_pair_scores(primary_scores, {p.key: p.benchmark_key for p in active}),
        "success_counts_scope": "仅本次公开已评分输入中依法采集的0/1；不含私测或未采集label",
        "ece": "not_computed",
        "capabilities": {
            "persistent_same_snapshot_reuse": True, "persistent_snapshot_extension": False,
            "real_data_loader": False, "custom_acquisition": False, "automatic_group_sampler": False,
            "parallel_workers": False, "candidate_container_isolation": False,
            "per_call_timeout": False, "per_response_visible_inputs": True,
            "summary_trace": True, "multi_file_submission": False, "ece": False,
        },
    }
