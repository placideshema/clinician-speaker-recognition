import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.speaker_identification import create_profile, save_profile
from src.web_app import (
    _jobs,
    _jobs_lock,
    parse_multipart,
    pipeline_progress,
    serialize_analysis,
    validate_upload,
)


class FakeAnnotation:
    def labels(self):
        return ["SPEAKER_00", "SPEAKER_01"]

    def __iter__(self):
        return iter(
            [
                (FakeTurn(0.2, 1.5), "SPEAKER_00"),
                (FakeTurn(1.6, 2.4), "SPEAKER_01"),
            ]
        )


class FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class FakeDiarization:
    speaker_diarization = FakeAnnotation()
    speaker_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


class WebAppTests(unittest.TestCase):
    def test_parse_multipart_extracts_fields_and_audio(self):
        boundary = "test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="clinician_id"\r\n\r\n'
            "clinician_001\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="audio"; filename="sample.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode() + b"wave-data\r\n" + f"--{boundary}--\r\n".encode()

        fields = parse_multipart(
            f"multipart/form-data; boundary={boundary}",
            body,
        )

        self.assertEqual(fields["clinician_id"], (None, b"clinician_001"))
        self.assertEqual(fields["audio"], ("sample.wav", b"wave-data"))

    def test_upload_requires_supported_audio_and_nonempty_content(self):
        self.assertEqual(validate_upload("recording.wav", b"audio"), ".wav")
        with self.assertRaisesRegex(ValueError, "Unsupported audio type"):
            validate_upload("recording.exe", b"audio")
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_upload("recording.wav", b"")

    def test_serializes_match_and_segments_without_marking_unknown_as_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "clinician.npz"
            save_profile(
                create_profile("clinician_001", [np.array([1.0, 0.0])]),
                profile_path,
            )
            result = serialize_analysis(FakeDiarization(), profile_path, 0.75)

        self.assertEqual(result["clinician_id"], "clinician_001")
        self.assertEqual(result["speakers"][0]["role"], "Clinician match")
        self.assertEqual(result["speakers"][1]["role"], "Unknown")
        self.assertFalse(result["unmatched_are_patient"])
        self.assertEqual(result["segments"][0]["start"], 0.2)
        self.assertEqual(result["duration"], 2.4)

    def test_pipeline_hook_updates_job_stage_and_monotonic_progress(self):
        job_id = "test-progress-job"
        with _jobs_lock:
            _jobs[job_id] = {"progress": 0}
        try:
            hook = pipeline_progress(job_id)
            hook("segmentation", None, completed=5, total=10)
            self.assertEqual(_jobs[job_id]["progress"], 36)
            self.assertEqual(
                _jobs[job_id]["phase"],
                "Detecting speech and speaker turns",
            )

            hook("embeddings", None, completed=1, total=4)
            self.assertEqual(_jobs[job_id]["progress"], 75)
            hook("segmentation", None, completed=1, total=10)
            self.assertEqual(_jobs[job_id]["progress"], 75)
        finally:
            with _jobs_lock:
                _jobs.pop(job_id, None)


if __name__ == "__main__":
    unittest.main()
