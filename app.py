import os
import uuid
import base64
import httpx
import io

from dotenv import load_dotenv

from PIL import Image, ImageOps

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# =========================================================
# CARREGA .ENV
# =========================================================

load_dotenv()

# =========================================================
# URL DA IA
# =========================================================

API_URL = "https://SEU-LINK.trycloudflare.com"

# =========================================================
# API KEY
# =========================================================

API_KEY = os.getenv("API_KEY")

# =========================================================
# PASTA OUTPUTS
# =========================================================

OUTPUT_DIR = r"C:\IA_OUTPUTS"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI()

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# STATIC
# =========================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# =========================================================
# CONFIGURAÇÕES IA
# =========================================================

TARGET_SIZE = (768, 768)

# =========================================================
# PREPARAR IMAGEM
# =========================================================

def preparar_imagem(image_bytes):

    # ABRIR IMAGEM
    img = Image.open(io.BytesIO(image_bytes))

    # RGB
    img = img.convert("RGB")

    # RESIZE PROFISSIONAL
    img = ImageOps.fit(
        img,
        TARGET_SIZE,
        method=Image.LANCZOS
    )

    # BUFFER
    buffer = io.BytesIO()

    # SALVAR
    img.save(
        buffer,
        format="PNG",
        quality=100
    )

    buffer.seek(0)

    return buffer.getvalue()

# =========================================================
# PREPARAR MÁSCARA
# =========================================================

def preparar_mask(mask_bytes):

    # ABRIR
    mask = Image.open(io.BytesIO(mask_bytes))

    # ESCALA DE CINZA
    mask = mask.convert("L")

    # RESIZE
    mask = ImageOps.fit(
        mask,
        TARGET_SIZE,
        method=Image.LANCZOS
    )

    # BUFFER
    buffer = io.BytesIO()

    # SALVAR
    mask.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return buffer.getvalue()

# =========================================================
# HOME
# =========================================================

@app.get("/")
async def home():

    return {
        "status": "API ONLINE"
    }

# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health():

    try:

        async with httpx.AsyncClient(timeout=30) as client:

            response = await client.get(
                f"{API_URL}/health"
            )

            return response.json()

    except Exception as e:

        print("\n======================================")
        print(" ERRO HEALTH ")
        print("======================================")

        print(str(e))

        return JSONResponse(
            status_code=500,
            content={
                "erro": str(e)
            }
        )

# =========================================================
# INPAINT
# =========================================================

@app.post("/api/inpaint")
async def inpaint(

    image: UploadFile = File(...),

    mask: UploadFile = File(...),

    prompt: str = Form(...),

    negative_prompt: str = Form(
        default="""
blurry,
low quality,
distorted,
deformed shoe,
melted textures,
duplicated parts,
extra laces,
broken sole,
unrealistic fabric,
noisy image,
bad proportions,
ugly stitching,
warped sneaker,
fake crochet texture,
oversaturated,
watermark,
cropped,
low resolution,
bad anatomy,
deformed,
mutated,
poor details,
artifact,
jpeg artifacts
"""
    ),

    strength: float = Form(default=0.55),

    guidance_scale: float = Form(default=7.5),

    steps: int = Form(default=35),

    seed: int = Form(default=-1),

    controlnet_scale: float = Form(default=1.0)

):

    try:

        # =================================================
        # VALIDAR IMAGENS
        # =================================================

        if not image.content_type.startswith("image/"):

            return JSONResponse(
                status_code=400,
                content={
                    "erro": "Arquivo enviado não é imagem"
                }
            )

        if not mask.content_type.startswith("image/"):

            return JSONResponse(
                status_code=400,
                content={
                    "erro": "Máscara enviada não é imagem"
                }
            )

        # =================================================
        # LER ARQUIVOS
        # =================================================

        image_bytes = await image.read()

        mask_bytes = await mask.read()

        # =================================================
        # PREPROCESSAMENTO IA
        # =================================================

        image_bytes = preparar_imagem(image_bytes)

        mask_bytes = preparar_mask(mask_bytes)

        # =================================================
        # ENVIAR PARA IA
        # =================================================

        async with httpx.AsyncClient(timeout=300) as client:

            response = await client.post(

                f"{API_URL}/v1/inpaint",

                headers={
                    "Authorization": f"Bearer {API_KEY}"
                },

                files={

                    "image": (
                        "image.png",
                        image_bytes,
                        "image/png"
                    ),

                    "mask": (
                        "mask.png",
                        mask_bytes,
                        "image/png"
                    ),
                },

                data={

                    "prompt": prompt,

                    "negative_prompt": negative_prompt,

                    "strength": strength,

                    "guidance_scale": guidance_scale,

                    "steps": steps,

                    "seed": seed,

                    "controlnet_scale": controlnet_scale
                }
            )

        # =================================================
        # STATUS
        # =================================================

        print("\n======================================")
        print(" STATUS ")
        print("======================================")

        print(response.status_code)

        # =================================================
        # VERIFICA ERRO
        # =================================================

        if response.status_code != 200:

            return JSONResponse(
                status_code=response.status_code,
                content={
                    "erro": "Erro na IA",
                    "resposta": response.text
                }
            )

        # =================================================
        # JSON
        # =================================================

        result = response.json()

        print("\n======================================")
        print(" RESPOSTA IA ")
        print("======================================")

        print(result)

        # =================================================
        # ENCONTRAR BASE64
        # =================================================

        def encontrar_base64(obj):

            # STRING
            if isinstance(obj, str):

                if obj.startswith("iVBOR"):
                    return obj

                if obj.startswith("/9j/"):
                    return obj

            # DICIONÁRIO
            if isinstance(obj, dict):

                for valor in obj.values():

                    resultado = encontrar_base64(valor)

                    if resultado:
                        return resultado

            # LISTA
            if isinstance(obj, list):

                for item in obj:

                    resultado = encontrar_base64(item)

                    if resultado:
                        return resultado

            return None

        # =================================================
        # BUSCAR IMAGEM
        # =================================================

        img_base64 = encontrar_base64(result)

        # =================================================
        # NÃO ENCONTROU
        # =================================================

        if not img_base64:

            return JSONResponse(
                status_code=500,
                content={
                    "erro": "Nenhuma imagem encontrada",
                    "resposta": result
                }
            )

        # =================================================
        # DECODIFICAR IMAGEM
        # =================================================

        image_data = base64.b64decode(img_base64)

        # =================================================
        # NOME ÚNICO
        # =================================================

        nome_arquivo = f"{uuid.uuid4()}.png"

        # =================================================
        # CAMINHO FINAL
        # =================================================

        caminho = os.path.join(
            OUTPUT_DIR,
            nome_arquivo
        )

        # =================================================
        # SALVAR IMAGEM
        # =================================================

        with open(caminho, "wb") as f:

            f.write(image_data)

        # =================================================
        # ABRIR AUTOMATICAMENTE
        # =================================================

        os.startfile(caminho)

        print("\n======================================")
        print(" IMAGEM SALVA ")
        print("======================================")

        print(caminho)

        # =================================================
        # RETORNO
        # =================================================

        return {

            "success": True,

            "arquivo": nome_arquivo,

            "caminho": caminho
        }

    except Exception as e:

        print("\n======================================")
        print(" ERRO ")
        print("======================================")

        print(str(e))

        return JSONResponse(
            status_code=500,
            content={
                "erro": str(e)
            }
        )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    print("\n======================================")
    print(" FASTAPI INICIADO ")
    print("======================================")

    print(f"\nAPI REMOTA: {API_URL}")

    print("\nPASTA OUTPUT:")
    print(OUTPUT_DIR)

    print("\nABRIR:")
    print("http://127.0.0.1:5001")
    print("http://127.0.0.1:5001/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5001,
        reload=True
    )