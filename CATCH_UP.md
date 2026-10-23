# Clinician Speaker Recognition — Catch-Up Guide

## 1. Project Goal

The goal of this project is to build a speaker recognition system for clinical conversations.

The initial goal is to reliably distinguish between two speakers:

- **Clinician**
- **Patient**

The longer-term goal is to identify the **specific clinician** speaking, rather than only labeling speakers as `SPEAKER_00` and `SPEAKER_01`.

The planned pipeline is:

```text
Clinical Audio
      ↓
Voice Activity Detection (VAD)
      ↓
Speaker Diarization
      ↓
Speaker Embeddings
      ↓
Known-Speaker Matching
      ↓
Clinician Identification
```

The current work is focused mainly on the **speaker diarization** stage.

---

# 2. Current Project Structure

```text
clinician-speaker-recognition/
│
├── data/
│   └── # Local audio files — NOT uploaded to GitHub
│
├── models/
│   └── # Local model files — NOT uploaded to GitHub
│
├── scripts/
│   └── enroll_clinician.py
│
├── src/
│   ├── diarization.py
│   ├── speaker_embedding.py
│   ├── speaker_identification.py
│   └── pipeline.py
│
├── .gitignore
├── requirements.txt
├── README.md
└── CATCH_UP.md
```

### Important

The `data/` and `models/` directories are intentionally excluded from GitHub.

Do **not** upload:

- Clinical recordings
- Patient recordings
- Other potentially sensitive audio
- Hugging Face tokens
- `.env` files
- The Python virtual environment

---

# 3. Python Environment

The project uses Python:

```text
Python 3.12.15
```

The environment is stored locally in:

```text
.venv/
```

The virtual environment is ignored by Git.

To create a new environment on another computer:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Then install the project dependencies:

```bash
pip install -r requirements.txt
```

Check Python:

```bash
python --version
```

Check pyannote:

```bash
python -c "import pyannote.audio; print(pyannote.audio.__version__)"
```

The version used during development was:

```text
pyannote.audio 4.0.7
```

---

# 4. Main Library: pyannote.audio

The main diarization library being used is:

```text
pyannote.audio
```

It was installed with:

```bash
pip install pyannote.audio
```

The project uses the pretrained model:

```text
pyannote/speaker-diarization-community-1
```

This model performs speaker diarization, meaning it determines:

1. When someone is speaking.
2. Which speech segments belong to the same speaker.

For example:

```text
0.00s → 4.20s : SPEAKER_00
4.20s → 8.50s : SPEAKER_01
8.50s → 12.10s : SPEAKER_00
```

At this stage, `SPEAKER_00` and `SPEAKER_01` do **not** mean "clinician" and "patient." They are anonymous speaker labels.

---

# 5. Hugging Face Setup

The pretrained Community-1 model is hosted on Hugging Face.

A Hugging Face account is required to access the model.

The model access conditions were accepted through the model page.

A Hugging Face access token was also created.

## Important security rule

The Hugging Face token must **never** be committed to GitHub or placed directly inside Python code.

The token was stored temporarily as an environment variable:

```bash
export HF_TOKEN="YOUR_TOKEN_HERE"
```

Check whether the token exists:

```bash
python -c "import os; print('HF token found:', bool(os.getenv('HF_TOKEN')))"
```

Expected:

```text
HF token found: True
```

Do not print the actual token.

If the token needs to be recreated, generate a new Hugging Face token rather than asking for the previous token.

---

# 6. Loading the Diarization Model

The model can be loaded with:

```python
import os
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"],
)
```

This was successfully tested.

The model downloaded its required components and loaded successfully.

Therefore, the following have already been confirmed:

- Python environment works.
- `pyannote.audio` works.
- Hugging Face authentication works.
- Community-1 access works.
- The pretrained diarization model loads locally.

---

# 7. Current Diarization Code

The main file is:

```text
src/diarization.py
```

The current implementation is approximately:

```python
import os
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"],
)

audio_file = "../data/clinical_audio.wav"

output = pipeline(audio_file, num_speakers=2)

print("\nDetected speaker segments:")

for turn, speaker in output.speaker_diarization:
    print(
        f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker}"
    )
```

The important part is:

```python
output = pipeline(audio_file, num_speakers=2)
```

Because the initial test case has two people, we tell the model that there are two speakers.

---

# 8. First Clinical-Style Test

A short two-person recording was created using Mac Voice Memos.

The recording was exported as:

```text
clinical_audio.m4a
```

and converted to:

```text
clinical_audio.wav
```

using:

```bash
ffmpeg -i ../data/clinical_audio.m4a \
    -ar 16000 \
    -ac 1 \
    ../data/clinical_audio.wav
```

The audio was converted to:

- 16 kHz
- Mono
- WAV

This is appropriate for the diarization model.

---

# 9. Problem With the First Test

When the short clinical-style recording was processed with:

```python
num_speakers=2
```

the result was approximately:

```text
0.96s → 10.04s : SPEAKER_01
10.04s → 18.21s : SPEAKER_00
```

This means the model divided the recording into two large sections rather than detecting the expected back-and-forth conversation.

When the model was run without specifying the number of speakers, it produced:

```text
0.96s → 18.21s : SPEAKER_00
```

So without the two-speaker constraint, it considered the entire short recording to be one speaker.

At this point, it was not clear whether the problem was:

- The recording quality
- The microphones/acoustics
- The short duration
- Speaker similarity
- Lack of clear pauses
- Crosstalk
- The recording setup

Importantly, this did **not** prove that pyannote was broken.

A separate test was needed.

---

# 10. AMI Meeting Corpus Test

To determine whether the problem was with pyannote or with the clinical-style test recording, a real multi-speaker dataset was tested.

The dataset used was the:

```text
AMI Meeting Corpus
```

A meeting recording was downloaded:

```text
ES2002a.Mix-Headset.wav
```

A 30-second clip was created:

```bash
ffmpeg -i ../data/ami_full.wav \
    -t 30 \
    ../data/ami_test_30s.wav
```

The 30-second clip was then processed using:

```python
output = pipeline(
    "../data/ami_test_30s.wav",
    num_speakers=2
)
```

---

# 11. AMI Test Result

The model detected multiple speaker turns:

```text
0.03s → 0.87s   : SPEAKER_01
4.77s → 7.89s   : SPEAKER_00
7.35s → 9.46s   : SPEAKER_01
8.59s → 9.13s   : SPEAKER_00
10.59s → 10.71s : SPEAKER_00
12.25s → 12.70s : SPEAKER_01
15.05s → 16.15s : SPEAKER_01
19.34s → 20.23s : SPEAKER_00
20.74s → 21.61s : SPEAKER_01
25.90s → 27.35s : SPEAKER_00
```

This was important because it demonstrated that:

**The pyannote installation and Community-1 model are working.**

The model was able to detect multiple speakers and speaker turns in real multi-speaker audio.

Some overlapping segments also appeared, which is possible in natural conversations.

---

# 12. What We Know Now

The following has been successfully completed:

### Environment

- Python 3.12.15 installed.
- Virtual environment created.
- `pyannote.audio 4.0.7` installed.
- FFmpeg installed and working.

### Hugging Face

- Hugging Face account configured.
- Community-1 model access granted.
- Local Hugging Face token configured.
- Community-1 successfully downloaded and loaded.

### Diarization

- `src/diarization.py` created.
- Two-speaker diarization tested.
- AMI real-world audio successfully tested.
- Multiple speaker turns detected.

### GitHub

The project has been pushed to GitHub.

The repository intentionally does **not** contain:

- `.venv`
- Clinical audio
- AMI audio
- Model files
- Hugging Face tokens

---

# 13. What Is NOT Finished

The most important unfinished problem is:

> The diarization system has not yet been validated reliably on the clinical-style two-speaker recording.

The AMI test shows that pyannote works, but the clinical recording needs additional testing.

We should **not move too quickly to speaker identification** until the diarization stage is reliable enough.

---

# 14. Recommended Next Steps

## Step 1 — Improve the two-speaker test

Create a cleaner test recording.

For example:

```text
Speaker A:
Hello, how are you feeling today?

Speaker B:
I am feeling better today.

Speaker A:
Have you experienced any pain?

Speaker B:
A little bit, especially in the morning.
```

The speakers should:

- Be physically separated if possible.
- Speak one at a time.
- Avoid talking over each other.
- Use a reasonably good microphone.
- Record for at least 1–2 minutes.

Then run the diarization again.

---

## Step 2 — Test different diarization settings

First:

```python
output = pipeline(
    audio_file,
    num_speakers=2
)
```

Then compare with:

```python
output = pipeline(audio_file)
```

Also consider testing:

```python
output = pipeline(
    audio_file,
    min_speakers=2,
    max_speakers=2
)
```

The goal is to determine which configuration produces the most realistic speaker turns.

---

# 15. Important Concept: Diarization vs Identification

These are two different problems.

### Diarization

Question:

> "Who spoke when?"

Output:

```text
0–5 sec   SPEAKER_00
5–9 sec   SPEAKER_01
9–14 sec  SPEAKER_00
```

The model does not know who these people actually are.

### Speaker Identification

Question:

> "Which known person is SPEAKER_00?"

For example:

```text
SPEAKER_00 → Dr. Smith
SPEAKER_01 → Patient
```

This requires speaker embeddings and enrolled speaker profiles.

---

# 16. Planned Speaker Recognition Stage

Once diarization is working reliably, the next major component will be speaker embeddings.

The planned library is:

```text
SpeechBrain
```

with an ECAPA-TDNN speaker recognition model.

The idea is:

```text
Speaker segment
      ↓
ECAPA-TDNN
      ↓
Speaker embedding
      ↓
Compare against enrolled clinician embeddings
      ↓
Best matching clinician
```

For example:

```text
Known clinician recordings

Dr. Smith:
    recording 1
    recording 2
    recording 3

        ↓

Generate embeddings

        ↓

Create Dr. Smith speaker profile
```

Then an unknown segment can be compared against that profile.

---

# 17. Planned Clinician Enrollment

The eventual system should have an enrollment process.

For example:

```text
scripts/enroll_clinician.py
```

Possible workflow:

```text
Clinician recording
       ↓
Diarization / speech segmentation
       ↓
SpeechBrain ECAPA-TDNN
       ↓
Embedding
       ↓
Save clinician profile
```

A clinician profile could eventually contain:

```text
clinician_id
embedding
metadata
```

The actual design is still to be decided.

---

# 18. Planned Speaker Identification

After enrollment:

```text
New clinical conversation
        ↓
Diarization
        ↓
SPEAKER_00
SPEAKER_01
        ↓
Extract embeddings
        ↓
Compare with enrolled clinicians
        ↓
Identify matching clinician
```

Potential output:

```text
0.0–4.2 sec   Dr. Smith
4.2–8.1 sec   Patient
8.1–12.4 sec  Dr. Smith
```

---

# 19. Files to Focus On

### `src/diarization.py`

Current priority.

This is where diarization is being tested.

### `src/speaker_embedding.py`

Currently planned for the next stage.

This will eventually contain the SpeechBrain/ECAPA-TDNN embedding logic.

### `src/speaker_identification.py`

Currently planned.

This will eventually compare an unknown speaker embedding against known clinician embeddings.

### `src/pipeline.py`

Eventually this should combine the entire workflow.

---

# 20. Important Security / Privacy Rules

This project involves clinical audio, so privacy needs to be taken seriously.

Never commit the following to GitHub:

```text
data/
models/
.env
HF tokens
clinical recordings
patient recordings
identifiable clinical information
```

The `.gitignore` already contains:

```text
.venv/
venv/
__pycache__/
*.pyc
.env
.DS_Store
data/
models/
```

If real patient data is eventually used, make sure the project has the appropriate institutional/ethical/privacy approval before processing it.

For development, use synthetic, de-identified, or publicly available datasets whenever possible.

---

# 21. Quick Start for the Next Developer

After cloning the repository:

```bash
git clone <repository-url>
cd clinician-speaker-recognition
```

Create the environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Hugging Face token:

```bash
export HF_TOKEN="YOUR_TOKEN"
```

Verify:

```bash
python -c "import os; print(bool(os.getenv('HF_TOKEN')))"
```

Then make sure a test audio file exists locally in:

```text
data/
```

For example:

```text
data/clinical_audio.wav
```

Run:

```bash
python src/diarization.py
```

---

# 22. Where to Continue

The most useful next task is:

> **Get reliable two-speaker diarization on a clean clinical-style recording.**

Do not start by training a new model.

First determine whether the issue is caused by the test recording.

Once diarization reliably produces something like:

```text
0.5s → 4.8s   SPEAKER_00
4.8s → 9.2s   SPEAKER_01
9.2s → 13.7s  SPEAKER_00
13.7s → 18.5s SPEAKER_01
```

then move to:

```text
Speaker embeddings
        ↓
Clinician enrollment
        ↓
Speaker identification
```

The current project is therefore at:

```text
[COMPLETED]
Python environment
       ↓
pyannote installation
       ↓
Hugging Face authentication
       ↓
Community-1 model loading
       ↓
Basic diarization
       ↓
AMI validation

[CURRENT]
Reliable two-speaker clinical diarization

[NEXT]
Speaker embeddings
       ↓
Clinician enrollment
       ↓
Clinician identification
       ↓
End-to-end pipeline
```