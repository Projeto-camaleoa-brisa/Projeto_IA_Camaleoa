import base64
import ast
import httpx

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
# ==================================================
# CONFIG
# ==================================================

API_URL = "https://acids-joshua-nathan-terrorists.trycloudflare.com/v1/inpaint"

API_KEY = os.getenv("API_KEY")

# ==================================================
# FASTAPI
# ==================================================

app = FastAPI()

# ==================================================
# STATIC
# ==================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ==================================================
# HOME
# ==================================================

@app.get("/")
async def home():

    return FileResponse("static/index.html")

# ==================================================
# HEALTH
# ==================================================

@app.get("/api/health")
async def health():

    return {
        "status": "ok",
        "api_url": API_URL
    }

# ==================================================
# FUNÇÃO PROCURA BASE64
# ==================================================

def encontrar_base64(obj):

    # string base64
    if isinstance(obj, str):

        if obj.startswith("iVBOR"):

            return obj

        if "base64," in obj:

            return obj.split(",")[1]

    # dicionário
    elif isinstance(obj, dict):

        for valor in obj.values():

            resultado = encontrar_base64(valor)

            if resultado:

                return resultado

    # lista
    elif isinstance(obj, list):

        for item in obj:

            resultado = encontrar_base64(item)

            if resultado:

                return resultado

    return None

# ==================================================
# INPAINT
# ==================================================

@app.post("/api/inpaint")
async def inpaint(

    image: UploadFile = File(...),

    mask: UploadFile = File(...),

    prompt: str = Form(
        default="luxury velvet high heel shoe, realistic velvet texture, premium fashion photography"
    ),

    negative_prompt: str = Form(
        default="blurry, low quality, distorted"
    ),

    strength: float = Form(
        default=0.75
    ),

    guidance_scale: float = Form(
        default=7.5
    ),

    steps: int = Form(
        default=30
    ),

    seed: int = Form(
        default=-1
    )

):

    try:

        # ==================================================
        # READ FILES
        # ==================================================

        image_bytes = await image.read()

        mask_bytes = await mask.read()

        # ==================================================
        # SAVE INPUTS
        # ==================================================

        with open("imagem.png", "wb") as f:

            f.write(image_bytes)

        with open("mask.png", "wb") as f:

            f.write(mask_bytes)

        # ==================================================
        # HEADERS
        # ==================================================

        headers = {

            "Authorization": f"Bearer {API_KEY}"

        }

        # ==================================================
        # FORM DATA
        # ==================================================

        data = {

            "prompt": prompt,

            "negative_prompt": negative_prompt,

            "strength": str(strength),

            "guidance_scale": str(guidance_scale),

            "steps": str(steps),

            "seed": str(seed)

        }

        # ==================================================
        # FILES
        # ==================================================

        files = {

            "image": (

                image.filename,

                image_bytes,

                image.content_type

            ),

            "mask": (

                mask.filename,

                mask_bytes,

                mask.content_type

            )

        }

        # ==================================================
        # REQUEST
        # ==================================================

        async with httpx.AsyncClient(timeout=300) as client:

            response = await client.post(

                API_URL,

                headers=headers,

                data=data,

                files=files

            )

        # ==================================================
        # DEBUG
        # ==================================================

        print("\n================ STATUS ================\n")

        print(response.status_code)

        print("\n================ RESPONSE TEXT ================\n")

        print(response.text)

        # ==================================================
        # ERROR HTTP
        # ==================================================

        if response.status_code != 200:

            return JSONResponse(

                status_code=response.status_code,

                content={

                    "error": response.text

                }

            )

        # ==================================================
        # PARSE RESPONSE
        # ==================================================

        try:

            result = response.json()

        except:

            result = ast.literal_eval(response.text)

        print("\n================ RESULT ================\n")

        print(result)

        # ==================================================
        # PROCURA BASE64
        # ==================================================

        img_base64 = encontrar_base64(result)

        # ==================================================
        # VALIDA
        # ==================================================

        if not img_base64:

            return JSONResponse(

                status_code=500,

                content={

                    "error": "Base64 nao encontrado",

                    "response": str(result)

                }

            )

        # ==================================================
        # DECODE
        # ==================================================

        img_bytes = base64.b64decode(img_base64)

        # ==================================================
        # SAVE RESULT
        # ==================================================

        with open("static/resultado.png", "wb") as f:

            f.write(img_bytes)

        print("\n================ IMAGE SAVED ================\n")

        # ==================================================
        # RETURN IMAGE
        # ==================================================

        return FileResponse("static/resultado.png")

    except Exception as e:

        print("\n================ EXCEPTION ================\n")

        print(str(e))

        return JSONResponse(

            status_code=500,

            content={

                "error": str(e)

            }

        )

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=5001,

        reload=True

    )