import os
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import HTTPException
from transformers import pipeline
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    print("✅ Groq API Key loaded successfully")
else:
    print("❌ GROQ_API_KEY not found in .env file")

client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1
)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not groq_api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment variables.")

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    if not file.filename.endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Unsupported file format. Use mp3, wav, or m4a.")

    try:
        session_id = uuid.uuid4().hex
        uploaded_path = os.path.join(AUDIO_DIR, f"{session_id}_{file.filename}")

        print("📥 Saving uploaded file...")
        with open(uploaded_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"✅ File saved: {uploaded_path}")
        print("🎧 Starting transcription...")
        result = asr_pipeline(uploaded_path, generate_kwargs={"language": "en"})
        transcription = result.get("text", "")

        print("📝 Transcription:", transcription)
        print("🧠 Sending transcription to Llama3...")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful recipe assistant. When asked for a recipe, respond with:\n\n"
                    "Ingredients:\n<list>\n\nSteps:\n1. step one\n2. step two\n..."
                )
            },
            {"role": "user", "content": transcription}
        ]

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages
        )
        llama3_response = response.choices[0].message.content
        print("🤖 Llama3 response:", llama3_response)

        print("🔊 Converting text to speech...")
        tts = gTTS(text=llama3_response, lang='en')
        audio_response_filename = f"{session_id}_response.mp3"
        audio_response_path = os.path.join(AUDIO_DIR, audio_response_filename)
        tts.save(audio_response_path)
        print(f"📁 gTTS audio saved: {audio_response_path}")

        # Cleanup
        file.file.close()
        os.remove(uploaded_path)

        return JSONResponse({
            "session_id": session_id,
            "transcription": transcription,
            "llama3_response": llama3_response,
            "download_url": f"/download-audio?filename={audio_response_filename}"
        })

    except Exception as e:
        print("❌ Error in /transcribe route:", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-audio")
async def download_audio(filename: str):
    try:
        file_path = os.path.join(AUDIO_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "Audio file not found."})

        return FileResponse(file_path, media_type="audio/mpeg", filename=filename)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
async def root():
    return {"message": "✅ FastAPI server is running"}
