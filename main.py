import os
import time
import uuid
import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- CONFIGURATION ---
os.environ["GOOGLE_API_KEY"] = "AIzaSyCJ6etqk3wTm3C-K9_5dpkfrYVK2oZSMTs"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI()


class AskRequest(BaseModel):
    video_url: str
    topic: str


class TimestampResponse(BaseModel):
    timestamp: str
    video_url: str
    topic: str


# Schema for Gemini Structured Output
class GeminiResponse(BaseModel):
    timestamp: str


def download_audio(url: str) -> str:
    """Downloads audio only using yt-dlp and returns the filename."""
    file_id = str(uuid.uuid4())
    output_filename = f"audio_{file_id}.mp3"

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'outtmpl': f"audio_{file_id}",  # yt-dlp adds extension automatically
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return output_filename


@app.post("/ask", response_model=TimestampResponse)
async def ask_gemini(request: AskRequest):
    audio_path = None
    try:
        # 1. Download Audio
        audio_path = download_audio(request.video_url)

        # 2. Upload to Gemini Files API
        audio_file = genai.upload_file(path=audio_path)

        # 3. Poll for ACTIVE state
        while audio_file.state.name == "PROCESSING":
            time.sleep(2)
            audio_file = genai.get_file(audio_file.name)

        if audio_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Audio processing failed")

        # 4. Generate Content with Structured Output
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Listen to this audio. Find the exact timestamp where the following topic or phrase is first spoken: '{request.topic}'. Return ONLY the timestamp in HH:MM:SS format."

        result = model.generate_content(
            [audio_file, prompt],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=GeminiResponse
            )
        )

        # Parse the structured response
        import json
        response_data = json.loads(result.text)
        timestamp = response_data.get("timestamp", "00:00:00")

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5. Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if 'audio_file' in locals():
            genai.delete_file(audio_file.name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
