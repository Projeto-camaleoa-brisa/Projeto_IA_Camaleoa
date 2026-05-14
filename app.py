import os
import base64
import httpx

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ====================================
# ENV
# ====================================

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

# ====================================
# FASTAPI
# ====================================

app = FastAPI()

# ====================================
# STATIC
# ====================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ====================================
# HOME
# ====================================

@app.get("/")
async def home():

    return FileResponse("static/index.html")

# ====================================
# HEALTH
# ====================================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "api_url": API_URL
    }

# ====================================
# INPAINT
# ====================================

@app.post("/api/inpaint")
async def inpaint(

    image: UploadFile = File(...),
    mask: UploadFile = File(...),

    prompt: str = Form(
        default="luxury red velvet sneaker"
    )

):

    try:

        image_bytes = await image.read()
        mask_bytes = await mask.read()

        with open("imagem.png", "wb") as f:
            f.write(image_bytes)

        with open("mask.png", "wb") as f:
            f.write(mask_bytes)

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "application/json"
        }

        files = {
            "image": (
                "imagem.png",
                open("imagem.png", "rb"),
                "image/png"
            ),

            "mask": (
                "mask.png",
                open("mask.png", "rb"),
                "image/png"
            )
        }

        data = {
            "prompt": prompt,
            "output_format": "png"
        }

        async with httpx.AsyncClient(timeout=120) as client:

            response = await client.post(
                f"{API_URL}/edit/inpaint",
                headers=headers,
                files=files,
                data=data
            )

        if response.status_code != 200:

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "error": response.text
                }
            )

        result = response.json()

        img_base64 = result["image"]

        img_bytes = base64.b64decode(img_base64)

        with open("static/resultado.png", "wb") as f:
            f.write(img_bytes)

        return FileResponse("static/resultado.png")

    except Exception as e:

        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )