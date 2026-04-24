import httpx
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

API_KEY = "sk-sd-colega1-asrtyuio2"
API_URL = "https://give-bell-republic-responses.trycloudflare.com"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{API_URL}/health")
            return r.json()
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})


@app.post("/api/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(default="red leather"),
    negative_prompt: str = Form(default="blurry, low quality, deformed"),
    strength: str = Form(default="0.75"),
    guidance_scale: str = Form(default="7.5"),
    steps: str = Form(default="30"),
    seed: str = Form(default="-1"),
):
    try:
        img_bytes = await image.read()
        mask_bytes = await mask.read()
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(
                f"{API_URL}/v1/inpaint",
                headers={"Authorization": f"Bearer {API_KEY}"},
                files={
                    "image": (image.filename, img_bytes, image.content_type),
                    "mask": (mask.filename, mask_bytes, mask.content_type),
                },
                data={
                    "prompt": prompt, "negative_prompt": negative_prompt,
                    "strength": strength, "guidance_scale": guidance_scale,
                    "steps": steps, "seed": seed,
                },
            )
            return r.json()
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": {"message": str(e)}})


if __name__ == "__main__":
    import uvicorn
    print(f"\n  API remota: {API_URL}")
    print(f"  Abre: http://localhost:5001\n")
    uvicorn.run(app, host="0.0.0.0", port=5001)