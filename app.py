# ================================
# IMPORTS
# ================================
import httpx
import base64
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

# ================================
# CONFIG
# ================================
load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

app = FastAPI()

print("API_URL carregada:", API_URL)
print("API_KEY carregada:", API_KEY[:5] + "*****" if API_KEY else None)

# ================================
# ROTAS BÁSICAS
# ================================
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/")
def index():
    return {"mensagem": "API rodando com Stable Diffusion 🚀"}

# ================================
# TESTE CONEXÃO COM IA
# ================================
@app.get("/health")
async def health_sd():
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{API_URL}/health")
        return {"status": "IA conectada", "response": r.status_code}
    except Exception as e:
        return JSONResponse(status_code=502, content={"erro": str(e)})

# ================================
# INPAINT
# ================================
@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form("croche texture"),
    negative_prompt: str = Form("low quality"),
    strength: float = Form(0.75),
    guidance_scale: float = Form(7.5),
    steps: int = Form(30),
    seed: int = Form(-1),
):
    try:
        # Ler arquivos
        img_bytes = await image.read()
        mask_bytes = await mask.read()

        # Requisição para API externa
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(
                url=f"{API_URL}/v1/inpaint",
                headers={
                    "Authorization": f"Bearer {API_KEY}"
                },
                files={
                    "image": (image.filename, img_bytes, image.content_type),
                    "mask": (mask.filename, mask_bytes, mask.content_type),
                },
                data={
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "strength": str(strength),
                    "guidance_scale": str(guidance_scale),
                    "steps": str(steps),
                    "seed": str(seed),
                },
            )

        # Processar resposta
        data = r.json()
        print("RESPOSTA DA IA:", data)

        # Converter base64 → imagem
        img_base64 = data["data"]["result_image_base64"]
        img_bytes = base64.b64decode(img_base64)

        # Salvar imagem
        with open("../static/outputs/resultado.png", "wb") as f:
            f.write(img_bytes)

        return {"mensagem": "Imagem salva como resultado.png"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

# ================================
# EXECUÇÃO
# ================================
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 API rodando em http://127.0.0.1:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)