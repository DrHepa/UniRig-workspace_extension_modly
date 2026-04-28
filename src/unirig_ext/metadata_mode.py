from __future__ import annotations

from typing import Literal


MetadataMode = Literal["auto", "legacy", "humanoid"]
VALID_METADATA_MODES: tuple[MetadataMode, ...] = ("auto", "legacy", "humanoid")
VALID_METADATA_MODES_LABEL = ", ".join(VALID_METADATA_MODES)


class MetadataModeError(ValueError):
    pass


def normalize_metadata_mode(params: dict[str, object]) -> MetadataMode:
    value = params.get("metadata_mode", "auto")
    if not isinstance(value, str):
        raise MetadataModeError(
            f"metadata_mode must be a string. Valid modes: {VALID_METADATA_MODES_LABEL}."
        )

    normalized = value.strip().lower()
    if not normalized:
        raise MetadataModeError(
            f"metadata_mode cannot be empty. Valid modes: {VALID_METADATA_MODES_LABEL}."
        )
    if normalized not in VALID_METADATA_MODES:
        raise MetadataModeError(
            f"unsupported metadata_mode {value!r}. Valid modes: {VALID_METADATA_MODES_LABEL}."
        )
    return normalized  # type: ignore[return-value]
