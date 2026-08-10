from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

app = FastAPI(title="AI Business Gateway")

OLLAMA_URL = "https://overrule-comma-darkish.ngrok-free.dev/api/generate"

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "Active", "message": "AI Engine is Online"}

@app.post("/generate")
async def generate_response(data: QueryRequest):
    try:
        payload = {
            "model": "llama3.2:1b",
            "prompt": data.prompt,
            "stream": False
        }
        res = requests.post(OLLAMA_URL, json=payload, timeout=60)
        
        if res.status_code == 200:
            return {"success": True, "response": res.json().get("response")}
        else:
            raise HTTPException(status_code=502, detail="Ollama engine error")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
