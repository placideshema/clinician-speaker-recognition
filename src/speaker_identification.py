from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.diarization import DIARIZATION_MODEL
from src.speaker_embedding import normalize_embedding

PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpeakerProfile:
    clinician_id: str
    embedding: NDArray[np.float32]
    model_id: str = DIARIZATION_MODEL


@dataclass(frozen=True)
class SpeakerMatch:
    speaker_label: str
    similarity: float
    matched: bool
    clinician_id: str


def validate_clinician_id(clinician_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", clinician_id):
        raise ValueError(
            "Clinician ID must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens."
        )
    if clinician_id in {".", ".."}:
        raise ValueError("Clinician ID cannot be a path component.")
    return clinician_id


def create_profile(
    clinician_id: str,
    embeddings: Sequence[NDArray[np.floating]],
    model_id: str = DIARIZATION_MODEL,
) -> SpeakerProfile:
    validate_clinician_id(clinician_id)
    if len(embeddings) == 0:
        raise ValueError("At least one enrollment embedding is required.")

    normalized = [normalize_embedding(embedding) for embedding in embeddings]
    dimensions = {embedding.size for embedding in normalized}
    if len(dimensions) != 1:
        raise ValueError("Enrollment embeddings must have matching dimensions.")

    centroid = normalize_embedding(np.mean(normalized, axis=0))
    return SpeakerProfile(clinician_id, centroid, model_id)


def save_profile(profile: SpeakerProfile, path: Path) -> None:
    validate_clinician_id(profile.clinician_id)
    path = path.expanduser().resolve()
    if path.suffix != ".npz":
        raise ValueError("Clinician profile files must use the .npz extension.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.array(PROFILE_SCHEMA_VERSION, dtype=np.int64),
        clinician_id=np.array(profile.clinician_id),
        model_id=np.array(profile.model_id),
        embedding=normalize_embedding(profile.embedding),
    )


def load_profile(path: Path) -> SpeakerProfile:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Clinician profile does not exist: {path}")

    with np.load(path, allow_pickle=False) as stored:
        required = {"schema_version", "clinician_id", "model_id", "embedding"}
        if not required.issubset(stored.files):
            missing = required - set(stored.files)
            raise ValueError(f"Profile is missing fields: {missing}")
        if int(stored["schema_version"]) != PROFILE_SCHEMA_VERSION:
            raise ValueError("Unsupported clinician profile schema version.")

        profile = SpeakerProfile(
            clinician_id=str(stored["clinician_id"].item()),
            embedding=normalize_embedding(stored["embedding"]),
            model_id=str(stored["model_id"].item()),
        )

    validate_clinician_id(profile.clinician_id)
    if profile.model_id != DIARIZATION_MODEL:
        raise ValueError(
            f"Profile uses {profile.model_id!r}; expected {DIARIZATION_MODEL!r}."
        )
    return profile


def match_speakers(
    speaker_embeddings: dict[str, NDArray[np.floating]],
    profile: SpeakerProfile,
    threshold: float,
) -> list[SpeakerMatch]:
    if not -1.0 <= threshold <= 1.0:
        raise ValueError("Cosine similarity threshold must be between -1 and 1.")

    enrollment = normalize_embedding(profile.embedding)
    scored = []
    for label, embedding in speaker_embeddings.items():
        candidate = normalize_embedding(embedding)
        if candidate.shape != enrollment.shape:
            raise ValueError(
                "Enrollment and diarization embeddings have different dimensions."
            )
        similarity = float(np.dot(enrollment, candidate))
        scored.append((label, similarity))

    above_threshold = [item for item in scored if item[1] >= threshold]
    unique_match = above_threshold[0][0] if len(above_threshold) == 1 else None
    return [
        SpeakerMatch(
            speaker_label=label,
            similarity=similarity,
            matched=label == unique_match,
            clinician_id=profile.clinician_id,
        )
        for label, similarity in scored
    ]