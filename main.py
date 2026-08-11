import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import HfApi

app = FastAPI(title="AI Business Gateway")

# استدعاء متغيرات البيئة من Render
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

# نموذج Hugging Face المجاني المستهدف للتوليد
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
HF_INFERENCE_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

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
        "inputs": request.prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }

    # 1. إرسال الطلب لـ Hugging Face Inference API
    try:
        response = requests.post(HF_INFERENCE_URL, headers=headers, json=payload, timeout=30)
        response_data = response.json()
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"HuggingFace API Error: {response_data}")

        # استخراج النص المولد
        if isinstance(response_data, list) and len(response_data) > 0:
            generated_text = response_data[0].get("generated_text", "")
        else:
            generated_text = str(response_data)

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to HuggingFace API: {str(e)}")

    # 2. حفظ النتيجة اختياري في مستودع البيانات السحابي (Dataset/Repo)
    upload_status = "skipped"
    if DATASET_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            log_content = f"Prompt: {request.prompt}\nResponse: {generated_text}\n"
            # حفظ العملية في ملف نصي داخل المستودع
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
