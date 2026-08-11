import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient, HfApi

app = FastAPI(title="AI Cloud Gateway with Telegram & Crypto")

# البيانات الأساسية المربوطة
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = os.getenv("DATASET_REPO")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8667895200:AAEko44gnEdx-fkNVkZvjFKLB6W1NYCBNgw")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "7205863758")
REDOTPAY_USDT_ADDRESS = os.getenv("REDOTPAY_USDT_ADDRESS", "TNP26wNYooiGZS7JCxpSbAZLGgJ56ffNSr")
CRYPTO_NETWORK = "TRC20"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def send_telegram_alert(message: str):
    """إرسال إشعار فوري لتيليجرام"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Telegram Notification Failed: {e}")

class QueryRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {
        "status": "online", 
        "message": "AI Gateway with Telegram & Crypto Payments is active!"
    }

@app.post("/generate")
def generate_response(request: QueryRequest):
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable is missing")

    # 1. إرسال إشعار باستلام طلب جديد
    send_telegram_alert(f"📥 *طلب جديد على السيرفر:*\n`{request.prompt}`")

    # 2. التوليد عبر الذكاء الاصطناعي
    try:
        client = InferenceClient(api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=256
        )
        generated_text = completion.choices[0].message.content

    except Exception as e:
        send_telegram_alert(f"⚠️ *خطأ أثناء معالجة الطلب:*\n`{str(e)}`")
        raise HTTPException(status_code=500, detail=f"HuggingFace Generation Error: {str(e)}")

    # 3. حفظ النتيجة سحابياً في Dataset
    upload_status = "skipped"
    if DATASET_REPO:
        try:
            api = HfApi(token=HF_TOKEN)
            log_bytes = f"Prompt: {request.prompt}\nResponse: {generated_text}\n".encode("utf-8")
            
            api.upload_file(
                path_or_fileobj=log_bytes,
                path_in_repo="logs/last_response.txt",
                repo_id=DATASET_REPO,
                repo_type="dataset"
            )
            upload_status = "success"
        except Exception as e:
            upload_status = f"failed: {str(e)}"

    # 4. إشعار بإتمام العملية
    send_telegram_alert("✅ *تمت معالجة الطلب وحفظه بنجاح!*")

    return {
        "status": "success",
        "prompt": request.prompt,
        "response": generated_text,
        "cloud_storage": upload_status
    }

@app.get("/payment-info")
def payment_info():
    """نقطة وصول الدفع المباشر بالكريبتو"""
    return {
        "currency": "USDT",
        "network": CRYPTO_NETWORK,
        "wallet_address": REDOTPAY_USDT_ADDRESS,
        "status": "active"
    }
