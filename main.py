import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient, HfApi

app = FastAPI(title="AI Business Gateway")

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

# استخدام نموذج Qwen المتاح مجاناً
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

    # 1. التوليد عبر مكتبة Hugging Face الرسمية
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=256
        )
        generated_text = completion.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HuggingFace Generation Error: {str(e)}")

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
