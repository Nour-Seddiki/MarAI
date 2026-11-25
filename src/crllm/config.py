"""Load configs/base.yaml + one overlay (+ CLI overrides) into typed dataclasses.

Unknown keys are an error, so a typo in a YAML file fails at load time instead of
silently falling back to a default.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import subprocess
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


@dataclass
class PathsCfg:
    data_dir: str = "data"
    ckpt_dir: str = "ckpt"
    reports_dir: str = "reports"
    constitution: str = "configs/constitution.yaml"


@dataclass
class ModelCfg:
    policy: str = "Qwen/Qwen3-0.6B-Base"
    reward: str = "Qwen/Qwen3-0.6B-Base"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    init_format_tokens: bool = True   # give untrained <think>/ChatML rows a distinct start (modeling.py)
    train_format_tokens: bool = True  # train those rows via PEFT trainable_token_indices


@dataclass
class TrainCfg:
    max_seq_len: int = 1024
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2.0e-4
    epochs: float = 2
    warmup_ratio: float = 0.03
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    logging_steps: int = 5
    save_steps: int = 100


@dataclass
class DataCfg:
    stage1_examples: int = 200
    stage2_prompts: int = 100
    stage3_prompts: int = 100
    stage3_k: int = 4
    stage3_min_rank_gap: int = 1
    rm_holdout_frac: float = 0.1
    eval_holdout_frac: float = 0.1


@dataclass
class GenerationCfg:
    backend: str = "hf"
    temperature: float = 0.9
    top_p: float = 0.95
    max_new_tokens: int = 768


@dataclass
class GRPOCfg:
    steps: int = 50
    num_generations: int = 4
    max_completion_length: int = 384
    reward_weights: list[float] = field(default_factory=lambda: [0.4, 0.1, 0.5])
    beta: float = 0.02
    use_vllm: bool = False
    temperature: float = 0.9


@dataclass
class JudgeCfg:
    base_url: str = "http://localhost:11434/v1"
    model: str = "qwen2.5:7b-instruct"
    api_key_env: str = "JUDGE_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 120


@dataclass
class EvalCfg:
    suites: list[str] = field(default_factory=lambda: ["format", "gsm8k", "redteam", "overrefusal", "honesty"])
    max_new_tokens: int = 768
    gsm8k_n: int = 1319


@dataclass
class TrackingCfg:
    report_to: str = "none"
    project: str = "crllm"


@dataclass
class HubCfg:
    repo: str | None = None
    private: bool = True


@dataclass
class RuntimeCfg:
    keep_awake: bool = False
    pin_p_cores: bool = False
    p_core_mask: int = 0x0FFF
    above_normal_priority: bool = False
    cuda_memory_fraction: float | None = None


@dataclass
class Config:
    seed: int = 42
    paths: PathsCfg = field(default_factory=PathsCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    lora: LoraCfg = field(default_factory=LoraCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    generation: GenerationCfg = field(default_factory=GenerationCfg)
    grpo: GRPOCfg = field(default_factory=GRPOCfg)
    judge: JudgeCfg = field(default_factory=JudgeCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)
    hub: HubCfg = field(default_factory=HubCfg)
    runtime: RuntimeCfg = field(default_factory=RuntimeCfg)
    name: str = "base"  # which overlay produced this config; set by load_config

    def path(self, key: str) -> Path:
        """Resolve one of `paths.*` against the project root."""
        return PROJECT_ROOT / getattr(self.paths, key)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------- merging


def deep_merge(base: dict, over: dict) -> dict:
    """Recursively merge `over` into a copy of `base`; dicts merge, everything else replaces."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(item: str) -> dict:
    """'train.lr=1e-4' -> {'train': {'lr': 1e-4}}; the value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like key.sub=value, got {item!r}")
    key, raw = item.split("=", 1)
    value = yaml.safe_load(raw) if raw != "" else ""
    if isinstance(value, str) and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)[eE][-+]?\d+", value):
        value = float(value)  # YAML 1.1 reads '1e-4' (no dot) as a string
    for part in reversed(key.strip().split(".")):
        value = {part: value}
    return value


# ---------------------------------------------------------------- typing


def _coerce(tp: Any, value: Any, where: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise TypeError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _build(tp, value, where)

    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors = []
        for a in args:
            if a is type(None):
                continue
            try:
                return _coerce(a, value, where)
            except (TypeError, ValueError) as e:
                errors.append(str(e))
        raise TypeError(f"{where}: {value!r} matches none of {tp} ({'; '.join(errors)})")

    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{where}: expected a list, got {value!r}")
        (item_tp,) = args or (Any,)
        return [_coerce(item_tp, v, f"{where}[{i}]") for i, v in enumerate(value)]

    if tp is Any:
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{where}: expected true/false, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{where}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise TypeError(f"{where}: expected a string, got {value!r}")
        return value
    raise TypeError(f"{where}: unsupported type {tp}")


def _build(cls: type, data: dict, where: str = "config"):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(names)}")
    kwargs = {k: _coerce(hints[k], v, f"{where}.{k}") for k, v in data.items()}
    return cls(**kwargs)


def _read_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping")
    return data


def load_config(overlay: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """base.yaml <- overlay (name like 'debug' or a path) <- 'a.b=c' overrides."""
    merged = _read_yaml(CONFIG_DIR / "base.yaml")
    name = "base"
    if overlay is not None:
        p = Path(overlay)
        if p.suffix not in (".yaml", ".yml"):
            p = CONFIG_DIR / f"{overlay}.yaml"
        elif not p.is_absolute() and not p.exists():
            p = PROJECT_ROOT / p
        merged = deep_merge(merged, _read_yaml(p))
        name = p.stem
    for item in overrides or []:
        merged = deep_merge(merged, parse_override(item))
    merged.pop("name", None)
    cfg = _build(Config, merged)
    cfg.name = name
    return cfg


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Shared CLI: --config debug|full|path.yaml and repeatable --set key.sub=value."""
    parser.add_argument("--config", default="debug", help="overlay name in configs/ or a YAML path")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    return load_config(args.config, args.overrides)


# ---------------------------------------------------------------- constitution


@dataclass(frozen=True)
class Principle:
    id: str
    text: str


@dataclass(frozen=True)
class Constitution:
    principles: tuple[Principle, ...]
    templates: dict[str, str]

    def get(self, principle_id: str) -> Principle:
        for p in self.principles:
            if p.id == principle_id:
                return p
        raise KeyError(f"no principle with id {principle_id!r}")

    def sample(self, rng: random.Random) -> Principle:
        return rng.choice(self.principles)

    def render(self, template: str, **fields: str) -> str:
        return self.templates[template].format(**fields)


TEMPLATE_FIELDS = {
    "critique": {"principle", "prompt", "response"},
    "revise": {"critique", "response"},
    "rank": {"principle", "prompt", "responses"},
}


def load_constitution(path: str | Path | None = None) -> Constitution:
    p = Path(path) if path else CONFIG_DIR / "constitution.yaml"
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    raw = _read_yaml(p)
    principles = tuple(Principle(id=d["id"], text=d["text"]) for d in raw["principles"])
    ids = [x.id for x in principles]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate principle ids in {p}: {ids}")
    templates = dict(raw["templates"])
    missing = set(TEMPLATE_FIELDS) - set(templates)
    if missing:
        raise KeyError(f"{p}: missing template(s) {sorted(missing)}")
    return Constitution(principles=principles, templates=templates)


# ---------------------------------------------------------------- reproducibility


def git_hash() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def run_metadata(cfg: Config, **extra: Any) -> dict[str, Any]:
    """What gets written as run.json next to every checkpoint."""
    return {"config": cfg.to_dict(), "seed": cfg.seed, "git": git_hash(), **extra}


def write_run_json(out_dir: str | Path, cfg: Config, **extra: Any) -> Path:
    out = Path(out_dir) / "run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run_metadata(cfg, **extra), indent=2), encoding="utf-8")
    return out
