from transformers import pipeline

# Use a more accurate model
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"  # use "cuda" if you have a GPU, else omit this line or use "cpu"
)

# Run the transcription
result = asr_pipeline("sandwiches.flac")

print(result["text"])
