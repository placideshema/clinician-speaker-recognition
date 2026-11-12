import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.speaker_embedding import extract_speaker_embeddings, normalize_embedding
from src.speaker_identification import (
    create_profile,
    load_profile,
    match_speakers,
    save_profile,
)


class FakeAnnotation:
    def labels(self):
        return ["SPEAKER_00", "SPEAKER_01"]


class FakeDiarizationOutput:
    speaker_diarization = FakeAnnotation()
    speaker_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class SpeakerEmbeddingTests(unittest.TestCase):
    def test_extracts_embeddings_in_annotation_label_order(self):
        embeddings = extract_speaker_embeddings(FakeDiarizationOutput())

        np.testing.assert_allclose(embeddings["SPEAKER_00"], [1.0, 0.0])
        np.testing.assert_allclose(embeddings["SPEAKER_01"], [0.0, 1.0])

    def test_rejects_zero_embedding(self):
        with self.assertRaisesRegex(ValueError, "zero vector"):
            normalize_embedding(np.zeros(2))


class SpeakerIdentificationTests(unittest.TestCase):
    def setUp(self):
        self.profile = create_profile("clinician_001", [np.array([1.0, 0.0])])

    def test_matches_single_speaker_above_threshold(self):
        matches = match_speakers(
            {
                "SPEAKER_00": np.array([0.98, 0.2]),
                "SPEAKER_01": np.array([0.0, 1.0]),
            },
            self.profile,
            threshold=0.75,
        )

        self.assertEqual(
            [(match.speaker_label, match.matched) for match in matches],
            [("SPEAKER_00", True), ("SPEAKER_01", False)],
        )

    def test_does_not_assign_clinician_if_multiple_speakers_pass(self):
        matches = match_speakers(
            {
                "SPEAKER_00": np.array([1.0, 0.0]),
                "SPEAKER_01": np.array([0.99, 0.1]),
            },
            self.profile,
            threshold=0.75,
        )

        self.assertTrue(all(not match.matched for match in matches))

    def test_saves_and_loads_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clinician.npz"
            save_profile(self.profile, path)
            loaded = load_profile(path)

        self.assertEqual(loaded.clinician_id, "clinician_001")
        self.assertEqual(loaded.model_id, self.profile.model_id)
        np.testing.assert_allclose(loaded.embedding, self.profile.embedding)


if __name__ == "__main__":
    unittest.main()
