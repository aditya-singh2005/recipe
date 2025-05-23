import os
import uuid
import re
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS
from transformers import pipeline

# Setup directories
BASE_DIR = os.path.dirname(__file__)
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load environment variables
load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))
groq_api_key = os.getenv("GROQ_API_KEY")
print(f"Groq API Key loaded: {'Yes' if groq_api_key else 'No - Key not found!'}")

# Initialize OpenAI (Groq) client
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

# Initialize Whisper ASR pipeline
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base",
    device="cpu"
)

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}

def whisper_transcribe(audio_path: str) -> str:
    result = asr_pipeline(audio_path)
    return result["text"]

def ask_llama(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful recipe assistant. When asked for a recipe, respond with:\n\nIngredients:\n<list>\n\nSteps:\n1. step one\n2. step two\n..."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq: {str(e)}"

def parse_recipe(text: str):
    ingredients = ""
    steps = []
    lines = text.split("\n")
    current_section = None
    for line in lines:
        line = line.strip()
        if line.lower().startswith("ingredients"):
            current_section = "ingredients"
            continue
        elif line.lower().startswith("steps"):
            current_section = "steps"
            continue
        if current_section == "ingredients" and line != "":
            ingredients += line + "\n"
        elif current_section == "steps" and line != "":
            if len(line) > 1 and line[0].isdigit() and line[1] in [".", ")"]:
                step_text = line[2:].strip()
            else:
                step_text = line
            steps.append(step_text)
    return ingredients.strip(), steps

def text_to_speech(text: str, filename: str):
    tts = gTTS(text,lang="en")
    full_path = os.path.join(AUDIO_DIR, filename)
    tts.save(full_path)

@app.get("/")
def read_root():
    return {"message": "Olive.ai is running!"}


@app.post("/recipe/start")
async def start_recipe(file: UploadFile = File(None), prompt: str = Form(None)):
    if file:
        audio_path = os.path.join(AUDIO_DIR, "temp_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        user_prompt = whisper_transcribe(audio_path)
    elif prompt:
        user_prompt = prompt
    else:
        return {"error": "No prompt or audio file provided"}

    full_recipe = ask_llama(user_prompt)
    ingredients, steps = parse_recipe(full_recipe)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "ingredients": ingredients,
        "steps": steps,
        "current_step": 0
    }

    intro_text = f"Here are the ingredients:\n{ingredients}\nShall I start with the first step?"
    audio_filename = f"audio_{session_id}_intro.mp3"
    text_to_speech(intro_text, audio_filename)

    return {
        "session_id": session_id,
        "intro_text": intro_text,
        "audio_url": f"/audio/{audio_filename}"
    }

@app.post("/recipe/next")
async def next_step(session_id: str = Form(...), file: UploadFile = File(None)):
    session = sessions.get(session_id)
    if not session:
        return {"error": "Invalid session ID"}

    if file:
        audio_path = os.path.join(AUDIO_DIR, "temp_next_audio.mp3")
        with open(audio_path, "wb") as f:
            f.write(await file.read())
        user_text = whisper_transcribe(audio_path).strip().lower()
    else:
        return {"error": "No audio file provided"}

    if not re.search(r'\bnext\b', user_text):
        prompt_text = "Please say 'next' to continue with the recipe steps."
        audio_filename = f"audio_{session_id}_say_next.mp3"
        text_to_speech(prompt_text, audio_filename)
        return {
            "step_text": prompt_text,
            "audio_url": f"/audio/{audio_filename}",
            "finished": False
        }

    idx = session["current_step"]
    steps = session["steps"]

    if idx >= len(steps):
        final_text = "You have completed all the steps. Enjoy your meal!"
        audio_filename = f"audio_{session_id}_done.mp3"
        text_to_speech(final_text, audio_filename)
        return {
            "step_text": final_text,
            "audio_url": f"/audio/{audio_filename}",
            "finished": True
        }

    step_text = steps[idx]
    session["current_step"] += 1
    audio_filename = f"audio_{session_id}_step_{idx+1}.mp3"
    text_to_speech(step_text, audio_filename)

    return {
        "step_text": step_text,
        "audio_url": f"/audio/{audio_filename}",
        "finished": False
    }

@app.get("/audio/{filename}")
def get_audio(filename: str):
    audio_path = os.path.join(AUDIO_DIR, filename)
    return FileResponse(audio_path, media_type="audio/mpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

