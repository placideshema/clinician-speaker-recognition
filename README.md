## Project Structure

```text
clinician-speaker-recognition/
│
├── src/
│   ├── diarization.py
│   ├── speaker_embedding.py
│   ├── speaker_identification.py
│   ├── pipeline.py
│   └── web_app.py
│
├── scripts/
│   └── enroll_clinician.py
├── web/
│   └── index.html
├── tests/
│   ├── test_speaker_identification.py
│   └── test_web_app.py
│
├── data/
│   └──              # Local only — NOT uploaded to GitHub
│
├── models/
│   └──              # Local only — NOT uploaded to GitHub
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Directory and File Description

- **`src/`** — Core speaker-recognition code.
  - `diarization.py` — Detects different speakers and determines when each speaker is talking.
  - `speaker_embedding.py` — Extracts speaker embeddings returned by pyannote.
  - `speaker_identification.py` — Compares speaker embeddings against enrolled clinicians and identifies the most likely clinician.
  - `pipeline.py` — Connects the complete workflow together.
  - `web_app.py` — Serves the local visualization interface and processing API.

- **`scripts/`** — Utility scripts for preparing and running the system.
  - `enroll_clinician.py` — Adds a known clinician's voice samples to the system so their voice can later be recognized.

- **`web/`** — The browser-based local interface.

- **`tests/`** — Focused unit tests for matching and web response formatting.

- **`data/`** — Local audio data, including clinical recordings and clinician enrollment samples. This directory must **not** be uploaded to GitHub.

- **`models/`** — Locally stored model files or generated speaker profiles. This directory must **not** be uploaded to GitHub.

- **`.gitignore`** — Prevents sensitive data, local environments, model files, and other unnecessary files from being committed to GitHub.

- **`requirements.txt`** — Lists the Python dependencies required to run the project.

- **`README.md`** — Documentation explaining the project, setup, architecture, and usage.

## Experimental clinician matching

The project can enroll one clinician voice and compare it with speakers found
by the pyannote diarization pipeline. Enroll only recordings containing the
clinician, with clear speech and as little background speech as possible. You
can provide multiple recordings; their normalized embeddings are averaged into
one local profile.

Set `HF_TOKEN` in your terminal and configure the Homebrew library paths used
by this macOS setup:

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/ffmpeg/lib"
```

Enroll the clinician using a pseudonymous ID:

```bash
python -m scripts.enroll_clinician \
  --clinician-id clinician_001 \
  --audio data/clinician_reference.wav
```

Add `--audio` again to enroll from multiple recordings. The profile is saved
under `models/clinician_profiles/`, which is git-ignored.

Then process a conversation:

```bash
python -m src.pipeline \
  --audio data/conversation.wav \
  --profile models/clinician_profiles/clinician_001.npz \
  --num-speakers 2
```

The initial cosine-similarity cutoff is `0.75` and is **experimental**, not
validated for clinical use. Change it with `--threshold` only for testing; a
cosine score is not a probability. If multiple diarized speakers pass the
cutoff, the pipeline deliberately avoids assigning the clinician to either.
Non-matching speakers remain `unknown`; the script does not automatically
classify them as patients.

The audio is read locally by pyannote after the model is downloaded. The
enrollment profile is derived biometric data: protect it like sensitive audio,
keep it out of source control, and process recordings only with appropriate
authorization. A successful match is experimental and must not be used as the
sole basis for clinical decisions.

## Local visual interface

With the virtual environment active and `HF_TOKEN` available in the terminal,
start the interface from the project directory:

```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib:/opt/homebrew/opt/ffmpeg/lib"
python -m src.web_app
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The interface lets you
enroll a clinician-only recording, choose a saved local profile, and analyze a
conversation. Results include per-speaker similarity scores and a timeline
aligned with audio playback. Enrollment and analysis run as background jobs;
the interface displays stage and progress updates while the page remains
responsive. The percentage is based on pyannote's segmentation and embedding
batch callbacks, with the remaining percentage reserved for final diarization
steps; model loading is shown as an initial indeterminate phase. Uploads are
sent only to this computer's localhost server, held temporarily while
processing, and removed afterward.
The profile remains in the git-ignored `models/clinician_profiles/` directory.
Stop the server with Ctrl+C. Do not expose this research interface to a
network or use its matches as clinical decisions.