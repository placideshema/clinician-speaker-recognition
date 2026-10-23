import os
from pyannote.audio import Pipeline

# Load the pretrained speaker diarization pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"],
)

# Audio file to process
audio_file = "../data/ami_test_30s.wav"

# We know this recording has two speakers:
# clinician + patient
output = pipeline(audio_file, num_speakers=2)

# Print every detected speaker segment
print("\nDetected speaker segments:")

for turn, speaker in output.speaker_diarization:
    print(
        f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker}"
    )