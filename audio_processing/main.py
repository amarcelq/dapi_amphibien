from fastapi import FastAPI
from pydantic import BaseModel
import requests
import asyncio
import audio_processing.src.main as audio

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}


class StartProcessRequest(BaseModel):
    path: str

@app.post("/start_process")
async def start_process(request: StartProcessRequest):
    # example sending progress
    session_key = request.session_key
    in_file_path = request.path
    request.post("web:8000/internal/progress/update/",json={"session_key":session_key,"progress":{"status":"running","name":"Loading File","description":"Loading the uploaded file"}})
    asyncio.create_task(process(session_key,in_file_path))
    return {"message": f"Process started for path: {in_file_path}"}



def process(session_key,in_file_path):
    audio