import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import HfApi

app = FastAPI(title="AI Business Gateway")

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

# الرابط الجديد والنموذج المدعوم مجاناً على Serverless Inference Router
HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "online", "message": "AI Gateway is up and running!"}

@app.post("/generate")
def generate_response(request: QueryRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable is missing")

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": request.prompt}
        ],
        "max_tokens": 256
    }

    # 1. الاتصال بـ Hugging Face API
    try:
        response = requests.post(HF_INFERENCE_URL, headers=headers, json=payload, timeout=30)
        response_data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"HuggingFace API Error: {response_data}")

        # استخراج النص المولد من استجابة Chat Completion
        generated_text = response_data['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to HuggingFace API: {str(e)}")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Unexpected response format: {response_data}")

    # 2. حفظ النتيجة سحابياً في Dataset
    upload_status = "skipped"
    if DATASET_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            log_content = f"Prompt: {request.prompt}\nResponse: {generated_text}\n"
            api.upload_file(
                path_or_bytes=log_content.encode("utf-8"),
                path_in_repo="logs/last_response.txt",
                repo_id=DATASET_REPO,
                repo_type="dataset"
            )
            upload_status = "success"
        except Exception as e:
            upload_status = f"failed: {str(e)}"

    return {
        "status": "success",
        "prompt": request.prompt,
        "response": generated_text,
        "cloud_storage": upload_status
    }
