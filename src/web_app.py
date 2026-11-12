import json
import logging
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
from urllib.parse import unquote, urlparse
import uuid

from src.diarization import (
    DIARIZATION_MODEL,
    diarize_with_pipeline,
    load_diarization_pipeline,
)
from src.speaker_embedding import extract_speaker_embeddings
from src.speaker_identification import (
    create_profile,
    load_profile,
    match_speakers,
    save_profile,
    validate_clinician_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIRECTORY = PROJECT_ROOT / "models" / "clinician_profiles"
WEB_DIRECTORY = PROJECT_ROOT / "web"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}

_pipeline = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = load_diarization_pipeline()
        return _pipeline


def get_profiles() -> list[dict[str, str]]:
    PROFILE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return [
        {"id": profile.stem, "filename": profile.name}
        for profile in sorted(PROFILE_DIRECTORY.glob("*.npz"))
    ]


def serialize_analysis(
    diarization_output,
    profile_path: Path,
    threshold: float,
) -> dict:
    profile = load_profile(profile_path)
    speaker_embeddings = extract_speaker_embeddings(diarization_output)
    matches = match_speakers(speaker_embeddings, profile, threshold)
    match_by_label = {match.speaker_label: match for match in matches}

    segments = [
        {
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        }
        for turn, speaker in diarization_output.speaker_diarization
    ]
    return {
        "clinician_id": profile.clinician_id,
        "threshold": threshold,
        "speakers": [
            {
                "label": match.speaker_label,
                "similarity": match.similarity,
                "matched": match.matched,
                "role": "Clinician match" if match.matched else "Unknown",
            }
            for match in matches
        ],
        "segments": segments,
        "duration": max((segment["end"] for segment in segments), default=0.0),
        "multiple_matches": sum(match.matched for match in matches) > 1,
        "unmatched_are_patient": False,
    }


def update_job(job_id: str, **updates) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            if "progress" in updates:
                updates["progress"] = max(
                    int(job["progress"]),
                    min(100, int(updates["progress"])),
                )
            job.update(updates)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def pipeline_progress(job_id: str):
    def report(step: str, _artifact=None, **details) -> None:
        completed = details.get("completed")
        total = details.get("total")
        if (
            isinstance(completed, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            fraction = max(0.0, min(1.0, completed / total))
        else:
            fraction = 0.0

        if step == "segmentation":
            progress = 8 + round(57 * fraction)
            message = "Detecting speech and speaker turns"
        elif step == "embeddings":
            progress = 68 + round(27 * fraction)
            message = "Comparing speaker voice embeddings"
        elif step == "speaker_counting":
            progress = 67
            message = "Estimating active speakers"
        elif step == "discrete_diarization":
            progress = 96
            message = "Finishing speaker timeline"
        else:
            return

        update_job(job_id, progress=progress, phase=message)

    return report


def run_job(
    job_id: str,
    task: str,
    suffix: str,
    audio_content: bytes,
    clinician_id: str | None = None,
    profile_path: Path | None = None,
    num_speakers: int | None = None,
    threshold: float | None = None,
) -> None:
    update_job(
        job_id,
        status="running",
        progress=3,
        phase="Loading diarization model",
    )
    try:
        pipeline = get_pipeline()
        update_job(job_id, progress=6, phase="Starting audio analysis")
        with tempfile.TemporaryDirectory(
            prefix="speaker-enrollment-" if task == "enroll" else "speaker-analysis-"
        ) as directory:
            audio_path = Path(directory) / f"audio{suffix}"
            audio_path.write_bytes(audio_content)
            with _inference_lock:
                output = diarize_with_pipeline(
                    pipeline,
                    audio_path,
                    num_speakers=1 if task == "enroll" else num_speakers,
                    progress_callback=pipeline_progress(job_id),
                )

            if task == "enroll":
                embeddings = extract_speaker_embeddings(output)
                if len(embeddings) != 1:
                    raise ValueError(
                        "Enrollment audio must contain one speaker. Use a clean "
                        "recording containing only the clinician."
                    )
                profile = create_profile(
                    clinician_id,
                    list(embeddings.values()),
                    model_id=DIARIZATION_MODEL,
                )
                output_path = PROFILE_DIRECTORY / f"{clinician_id}.npz"
                save_profile(profile, output_path)
                result = {
                    "clinician_id": clinician_id,
                    "profile": output_path.name,
                    "message": "Clinician voice profile saved locally.",
                }
            else:
                result = serialize_analysis(output, profile_path, threshold)

        update_job(
            job_id,
            status="completed",
            progress=100,
            phase="Complete",
            result=result,
        )
    except Exception as error:
        logging.exception("Local speaker recognition job failed")
        update_job(
            job_id,
            status="failed",
            phase="Failed",
            error=str(error) or "Processing failed.",
        )


def start_job(task: str, **parameters) -> str:
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "task": task,
            "status": "queued",
            "progress": 0,
            "phase": "Queued",
        }

    thread = threading.Thread(
        target=run_job,
        args=(
            job_id,
            task,
            parameters["suffix"],
            parameters["audio_content"],
            parameters.get("clinician_id"),
            parameters.get("profile_path"),
            parameters.get("num_speakers"),
            parameters.get("threshold"),
        ),
        daemon=True,
    )
    thread.start()
    return job_id


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: "
        + content_type.encode("ascii")
        + b"\r\nMIME-Version: 1.0\r\n\r\n"
        + body
    )
    if not message.is_multipart():
        raise ValueError("Expected multipart form data.")

    fields = {}
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        name = part.get_param("name", header="content-disposition")
        if not name or not disposition.startswith("form-data"):
            continue
        filename = part.get_filename()
        fields[name] = (filename, part.get_payload(decode=True) or b"")
    return fields


def validate_upload(filename: str | None, content: bytes) -> str:
    if not filename:
        raise ValueError("Choose an audio file.")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            "Unsupported audio type. Use WAV, MP3, M4A, FLAC, OGG, or AAC."
        )
    if not content:
        raise ValueError("The uploaded audio file is empty.")
    return suffix


class SpeakerRecognitionHandler(BaseHTTPRequestHandler):
    server_version = "LocalSpeakerRecognition/1.0"

    def _is_local_request(self) -> bool:
        host = self.headers.get("Host", "").lower()
        allowed_hosts = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if host not in allowed_hosts:
            return False

        origin = self.headers.get("Origin")
        if origin is None:
            return True
        parsed_origin = urlparse(origin)
        return (
            parsed_origin.scheme == "http"
            and parsed_origin.netloc.lower() in allowed_hosts
        )

    def do_GET(self) -> None:
        if not self._is_local_request():
            self._send_json(403, {"error": "This interface is restricted to localhost."})
            return
        path = urlparse(self.path).path
        if path == "/":
            page = WEB_DIRECTORY / "index.html"
            self._send_bytes(
                200,
                page.read_bytes(),
                "text/html; charset=utf-8",
                extra_headers={
                    "Content-Security-Policy": (
                        "default-src 'self' blob: data:; "
                        "script-src 'self' 'unsafe-inline'; "
                        "style-src 'self' 'unsafe-inline'; "
                        "connect-src 'self';"
                    ),
                    "Cache-Control": "no-store",
                },
            )
        elif path == "/api/profiles":
            self._send_json(200, {"profiles": get_profiles()})
        elif path.startswith("/api/jobs/"):
            job_id = unquote(path.removeprefix("/api/jobs/"))
            try:
                uuid.UUID(hex=job_id)
            except ValueError:
                self._send_json(404, {"error": "Job not found."})
                return
            job = get_job(job_id)
            if job is None:
                self._send_json(404, {"error": "Job not found."})
                return
            self._send_json(200, job)
        else:
            self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        if not self._is_local_request():
            self._send_json(403, {"error": "This interface is restricted to localhost."})
            return
        route = urlparse(self.path).path
        if route not in {"/api/enroll", "/api/analyze"}:
            self._send_json(404, {"error": "Not found."})
            return

        try:
            fields = self._read_form()
            if route == "/api/enroll":
                job_id = self._queue_enrollment(fields)
            else:
                job_id = self._queue_analysis(fields)
            self._send_json(202, {"job_id": job_id})
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
        except Exception as error:
            logging.exception("Local speaker recognition request failed")
            self._send_json(500, {"error": str(error) or "Processing failed."})

    def _read_form(self) -> dict[str, tuple[str | None, bytes]]:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise ValueError("Submit the form as multipart form data.")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid request size.") from error
        if content_length <= 0:
            raise ValueError("The request is empty.")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError("Audio uploads must be smaller than 250 MB.")
        body = self.rfile.read(content_length)
        if len(body) != content_length:
            raise ValueError("The audio upload was incomplete.")
        return parse_multipart(content_type, body)

    @staticmethod
    def _required_text(fields, key: str) -> str:
        value = fields.get(key)
        if value is None:
            raise ValueError(f"Missing form field: {key}.")
        text = value[1].decode("utf-8").strip()
        if not text:
            raise ValueError(f"Missing form field: {key}.")
        return text

    @staticmethod
    def _required_audio(fields) -> tuple[str, bytes]:
        filename, content = fields.get("audio", (None, b""))
        suffix = validate_upload(filename, content)
        return suffix, content

    def _queue_enrollment(self, fields) -> str:
        clinician_id = validate_clinician_id(
            self._required_text(fields, "clinician_id")
        )
        suffix, content = self._required_audio(fields)
        return start_job(
            "enroll",
            clinician_id=clinician_id,
            suffix=suffix,
            audio_content=content,
        )

    def _queue_analysis(self, fields) -> str:
        profile_name = self._required_text(fields, "profile")
        profile_path = (PROFILE_DIRECTORY / profile_name).resolve()
        if profile_path.parent != PROFILE_DIRECTORY.resolve():
            raise ValueError("Invalid clinician profile selection.")
        if profile_path.suffix != ".npz" or not profile_path.is_file():
            raise ValueError("Select an existing clinician profile.")

        try:
            num_speakers = int(self._required_text(fields, "num_speakers"))
            threshold = float(self._required_text(fields, "threshold"))
        except ValueError as error:
            raise ValueError("Speaker count and threshold must be valid numbers.") from error
        if num_speakers < 1:
            raise ValueError("Speaker count must be at least one.")
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between -1 and 1.")

        suffix, content = self._required_audio(fields)
        return start_job(
            "analyze",
            suffix=suffix,
            audio_content=content,
            profile_path=profile_path,
            num_speakers=num_speakers,
            threshold=threshold,
        )

    def _send_json(self, status: int, payload: dict) -> None:
        self._send_bytes(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
            extra_headers={"Cache-Control": "no-store"},
        )

    def _send_bytes(
        self,
        status: int,
        content: bytes,
        content_type: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        logging.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the local speaker recognition visualization."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("For privacy, the web app may only bind to localhost.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    server = ThreadingHTTPServer((args.host, args.port), SpeakerRecognitionHandler)
    print(f"Local speaker recognition UI: http://{args.host}:{args.port}")
    print("Audio is processed locally and temporary uploads are removed afterward.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping local speaker recognition UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
