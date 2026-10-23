## Project Structure

```text
clinician-speaker-recognition/
│
├── src/
│   ├── diarization.py
│   ├── speaker_embedding.py
│   ├── speaker_identification.py
│   └── pipeline.py
│
├── scripts/
│   └── enroll_clinician.py
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
  - `speaker_embedding.py` — Generates speaker embeddings from audio using a pretrained speaker-recognition model.
  - `speaker_identification.py` — Compares speaker embeddings against enrolled clinicians and identifies the most likely clinician.
  - `pipeline.py` — Connects the complete workflow together.

- **`scripts/`** — Utility scripts for preparing and running the system.
  - `enroll_clinician.py` — Adds a known clinician's voice samples to the system so their voice can later be recognized.

- **`data/`** — Local audio data, including clinical recordings and clinician enrollment samples. This directory must **not** be uploaded to GitHub.

- **`models/`** — Locally stored model files or generated speaker profiles. This directory must **not** be uploaded to GitHub.

- **`.gitignore`** — Prevents sensitive data, local environments, model files, and other unnecessary files from being committed to GitHub.

- **`requirements.txt`** — Lists the Python dependencies required to run the project.

- **`README.md`** — Documentation explaining the project, setup, architecture, and usage.