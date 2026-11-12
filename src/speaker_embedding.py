from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray


class SpeakerAnnotation(Protocol):
    def labels(self) -> Sequence[str]: ...


class DiarizationOutput(Protocol):
    speaker_diarization: SpeakerAnnotation
    speaker_embeddings: NDArray[np.floating] | None


def normalize_embedding(embedding: NDArray[np.floating]) -> NDArray[np.float32]:
    vector = np.asarray(embedding, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(
            "A speaker embedding must be a non-empty one-dimensional vector."
        )
    if not np.isfinite(vector).all():
        raise ValueError("A speaker embedding contains non-finite values.")

    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("A speaker embedding cannot be a zero vector.")
    return vector / norm


def extract_speaker_embeddings(
    output: DiarizationOutput,
) -> dict[str, NDArray[np.float32]]:
    embeddings = output.speaker_embeddings
    if embeddings is None:
        raise ValueError("The diarization pipeline did not return speaker embeddings.")

    labels = list(output.speaker_diarization.labels())
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(labels):
        raise ValueError(
            "Diarization labels and speaker embedding rows do not match."
        )

    return {
        label: normalize_embedding(embeddings[index])
        for index, label in enumerate(labels)
    }