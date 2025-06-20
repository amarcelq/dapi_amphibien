from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/healthy")
async def healthy():
    return {"message": "Alive"}


class StartProcessRequest(BaseModel):
    path: str

@app.post("/start_process")
async def start_process(request: StartProcessRequest):
    # Here you can add your process starting logic using request.path
    return {"message": f"Process started for path: {request.path}"}