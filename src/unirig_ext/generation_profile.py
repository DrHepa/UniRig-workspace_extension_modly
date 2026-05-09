from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bootstrap import RuntimeContext


ARTICULATIONXL_PROFILE = "articulationxl"
VROID_PROFILE = "vroid"
ALLOWED_GENERATION_PROFILES = (ARTICULATIONXL_PROFILE, VROID_PROFILE)
REJECTED_GENERATION_PASSTHROUGH_KEYS = frozenset(
    {
        "task",
        "class",
        "cls",
        "yaml",
        "config",
        "generation_kwargs",
        "generate_kwargs",
    }
)
ARTICULATIONXL_SKELETON_TASK = "configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml"
VROID_GENERATED_CONFIG_RELATIVE = Path("generation_profiles") / "vroid_skeleton_task.yaml"


class GenerationProfileValidationError(ValueError):
    pass


class GenerationProfileConfigError(ValueError):
    def __init__(self, *, profile: str, key: str, path: Path, message: str | None = None) -> None:
        self.profile = profile
        self.key = key
        self.path = path
        detail = message or f"missing required upstream key '{key}'"
        super().__init__(
            "generation_profile profile-configuration error: "
            f"profile={profile}; key={key}; path={path}; {detail}"
        )


@dataclass(frozen=True)
class GenerationProfile:
    name: str
    status: str
    skeleton_prior: str
    skeleton_task: str
    profile_config_source: str
    generated_config_path: Path | None = None
    generated_config_sha256: str | None = None

    @property
    def config_source(self) -> str:
        return self.profile_config_source

    @property
    def config_path(self) -> Path | None:
        return self.generated_config_path

    @property
    def config_sha256(self) -> str | None:
        return self.generated_config_sha256


def normalize_generation_profile(params: dict[str, Any]) -> GenerationProfile:
    rejected = sorted(key for key in params if key in REJECTED_GENERATION_PASSTHROUGH_KEYS)
    if rejected:
        raise GenerationProfileValidationError(
            "unsupported generation profile public field(s): "
            f"{', '.join(rejected)}. Use generation_profile with allowed values: "
            f"{', '.join(ALLOWED_GENERATION_PROFILES)}."
        )

    value = params.get("generation_profile", ARTICULATIONXL_PROFILE)
    if not isinstance(value, str):
        raise GenerationProfileValidationError(
            "generation_profile must be a string with one of: "
            f"{', '.join(ALLOWED_GENERATION_PROFILES)}."
        )
    normalized = value.strip().lower()
    if normalized not in ALLOWED_GENERATION_PROFILES:
        raise GenerationProfileValidationError(
            "generation_profile must be one of: "
            f"{', '.join(ALLOWED_GENERATION_PROFILES)}. Received: {value!r}."
        )
    if normalized == VROID_PROFILE:
        return GenerationProfile(
            name=VROID_PROFILE,
            status="experimental",
            skeleton_prior=VROID_PROFILE,
            skeleton_task="",
            profile_config_source="generated_run_config",
        )
    return _articulationxl_profile()


def resolve_generation_profile(profile: GenerationProfile, *, context: RuntimeContext, run_dir: Path) -> GenerationProfile:
    if profile.name == ARTICULATIONXL_PROFILE:
        return _articulationxl_profile()
    if profile.name == VROID_PROFILE:
        return _resolve_vroid_profile(context=context, run_dir=run_dir)
    raise GenerationProfileValidationError(
        "generation_profile must be one of: "
        f"{', '.join(ALLOWED_GENERATION_PROFILES)}. Received: {profile.name!r}."
    )


def sidecar_diagnostics(profile: GenerationProfile) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "skeleton_prior": profile.skeleton_prior,
        "profile_config_source": profile.profile_config_source,
        "trust_effect": "none",
    }
    if profile.generated_config_path is not None:
        diagnostics["generated_config_path"] = str(profile.generated_config_path)
    if profile.generated_config_sha256 is not None:
        diagnostics["generated_config_sha256"] = profile.generated_config_sha256
    return diagnostics


def _articulationxl_profile() -> GenerationProfile:
    return GenerationProfile(
        name=ARTICULATIONXL_PROFILE,
        status="stable",
        skeleton_prior=ARTICULATIONXL_PROFILE,
        skeleton_task=ARTICULATIONXL_SKELETON_TASK,
        profile_config_source="upstream_task",
    )


def _resolve_vroid_profile(*, context: RuntimeContext, run_dir: Path) -> GenerationProfile:
    source_path = context.unirig_dir / ARTICULATIONXL_SKELETON_TASK
    source = _load_upstream_config(source_path, profile=VROID_PROFILE)
    generated = copy.deepcopy(source)
    _require_path(generated, "task", source_path)
    _require_path(generated, "system.skeleton_prior", source_path)
    _require_path(generated, "generate_kwargs", source_path)
    _require_path(generated, "tokenizer.skeleton_order", source_path)

    task = generated["task"]
    if isinstance(task, dict):
        task["assign_cls"] = VROID_PROFILE
    generated["system"]["skeleton_prior"] = VROID_PROFILE
    if isinstance(generated["generate_kwargs"], dict):
        generated["generate_kwargs"]["cls"] = VROID_PROFILE
    generated["tokenizer"]["skeleton_prior"] = VROID_PROFILE

    rendered = json.dumps(generated, indent=2, sort_keys=True) + "\n"
    destination = run_dir / VROID_GENERATED_CONFIG_RELATIVE
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return GenerationProfile(
        name=VROID_PROFILE,
        status="experimental",
        skeleton_prior=VROID_PROFILE,
        skeleton_task=str(destination),
        profile_config_source="generated_run_config",
        generated_config_path=destination,
        generated_config_sha256=digest,
    )


def _load_upstream_config(path: Path, *, profile: str) -> dict[str, Any]:
    if not path.exists():
        raise GenerationProfileConfigError(profile=profile, key="upstream_task", path=path, message="upstream task config is missing")
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as import_exc:
            raise GenerationProfileConfigError(
                profile=profile,
                key="upstream_task",
                path=path,
                message=(
                    "upstream task config is not JSON and PyYAML is unavailable for controlled YAML seam parsing: "
                    f"{json_exc}"
                ),
            ) from import_exc
        try:
            loaded = yaml.safe_load(text)
        except Exception as exc:
            raise GenerationProfileConfigError(
                profile=profile,
                key="upstream_task",
                path=path,
                message=f"upstream task config is not parseable by the controlled profile resolver: {exc}",
            ) from exc
    except OSError as exc:
        raise GenerationProfileConfigError(
            profile=profile,
            key="upstream_task",
            path=path,
            message=f"upstream task config cannot be read by the controlled profile resolver: {exc}",
        ) from exc
    if not isinstance(loaded, dict):
        raise GenerationProfileConfigError(profile=profile, key="upstream_task", path=path, message="upstream task config must be an object")
    return loaded


def _require_path(config: dict[str, Any], dotted_key: str, path: Path) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise GenerationProfileConfigError(profile=VROID_PROFILE, key=dotted_key, path=path)
        current = current[part]
    return current
